using OhMyREPL
using ForwardDiff
using ReverseDiff
using Optim
using LinearAlgebra
using Plots

include("lennard-jones.jl")

const xyzscale = 2.0

function print_xyz(io, r)
    println(io, size(r, 2))
    println(io)
    for c in eachcol(r)
        println(io, "H    ",
            c[1] * xyzscale, " ",
            c[2] * xyzscale, " ",
            c[3] * xyzscale)
    end
end

function print_xyz(io, r, v, i, t, Δt, V, K, printerval)
    println(io, size(r, 2))
    println(io,
        "i = ", i,
        "; t = ", t,
        "; Δt = ", Δt,
        "; V = ", V,
        "; K = ", K,
        "; E = ", V + K,
        "; printerval = ", printerval)
    for (rc, vc) in zip(eachcol(r), eachcol(v))
        println(io, "H    ",
            rc[1] * xyzscale, ' ',
            rc[2] * xyzscale, ' ',
            rc[3] * xyzscale, ' ',
            vc[1] * xyzscale, ' ',
            vc[2] * xyzscale, ' ',
            vc[3] * xyzscale)
    end
end

function setup_cubic_lattice(n, r)
    coords = Float64[]

    is = 0:(n-1)
    for i in is, j in is, k in is
        append!(coords, (i, j, k) .* r)
    end

    reshape(coords, 3, n^3)
end

function make_rd_grad(pot, r)
    tape = ReverseDiff.GradientTape(pot, r)
    comp_tape = ReverseDiff.compile(tape)

    function g!(g, r)
        ReverseDiff.gradient!(g, comp_tape, r)
    end
end

function make_egf(r_eq, E_b, r)
    pot = lennard_jones_par(r_eq, E_b)
    g! = lennard_jones_grad_par(r_eq, E_b, r)
    function egf!(g, r)
        g!(g, r)
        pot(r)
    end
end

function calc_kin_e(v)
    sum(0.5 * (vc ⋅ vc) for vc in eachcol(v))
end

function do_md(io::IO, n_steps, Δt, egf!, r, v=zeros(size(r));
    add_first=true, t0=0.0, printerval=1)

    g = similar(r)

    V = egf!(g, r)
    K = calc_kin_e(v)

    t = t0

    if add_first
        print_xyz(io, r, v, 0, t, Δt, V, K, printerval)
    end

    timer = time()

    for i in 1:((n_steps÷printerval)*printerval)
        if time() - timer > 1.0
            timer += 1.0
            println(i, "/", n_steps, "\t\t",
                round((i / n_steps) * 100; digits=2), "%")
        end

        axpy!(-0.5 * Δt, g, v)
        axpy!(Δt, v, r)
        V = egf!(g, r)
        axpy!(-0.5 * Δt, g, v)

        t += Δt

        if i % printerval == 0
            K = calc_kin_e(v)
            print_xyz(io, r, v, i, t, Δt, V, K, printerval)
        end

        # if isfile("stop")
        #     break
        # end
    end
end

function get_last_conf(filename)
    t = 0.0
    Δt = 0.0
    printerval = 1

    r = Float64[]
    v = Float64[]

    open(filename) do io
        lines = Iterators.Stateful(eachline(io))

        while !isempty(lines)
            r = Float64[]
            v = Float64[]

            n_atm = parse(Int, popfirst!(lines))
            l = popfirst!(lines)
            ls = split(l, "; ")

            t = parse(Float64, split(ls[2], " = ")[2])
            Δt = parse(Float64, split(ls[3], " = ")[2])
            printerval = parse(Int, split(ls[7], " = ")[2])

            for _ in 1:n_atm
                l = popfirst!(lines)
                ls = split(l)
                append!(r, parse(Float64, rc) for rc in ls[2:4])
                append!(v, parse(Float64, vc) for vc in ls[5:7])
            end
        end
    end

    reshape(r, 3, length(r) ÷ 3) / xyzscale,
    reshape(v, 3, length(v) ÷ 3) / xyzscale,
    t,
    Δt,
    printerval
end

function get_nth_conf(filename, n)
    t = 0.0
    Δt = 0.0
    printerval = 1

    r = Float64[]
    v = Float64[]

    open(filename) do io
        lines = Iterators.Stateful(eachline(io))

        i = 0
        while !isempty(lines)
            r = Float64[]
            v = Float64[]

            n_atm = parse(Int, popfirst!(lines))
            l = popfirst!(lines)
            ls = split(l, "; ")

            t = parse(Float64, split(ls[2], " = ")[2])
            Δt = parse(Float64, split(ls[3], " = ")[2])
            printerval = parse(Int, split(ls[7], " = ")[2])

            for _ in 1:n_atm
                l = popfirst!(lines)
                ls = split(l)
                append!(r, parse(Float64, rc) for rc in ls[2:4])
                append!(v, parse(Float64, vc) for vc in ls[5:7])
            end

            if i == n
                break
            end
    
            i += 1
        end
    end

    reshape(r, 3, length(r) ÷ 3) / xyzscale,
    reshape(v, 3, length(v) ÷ 3) / xyzscale,
    t,
    Δt,
    printerval
end

function resume_md(filename, pot, n_steps;
    Δt=nothing, v_scale=1.0, printerval=nothing)
    r, v, t, Δt_l, printerval_l = get_last_conf(filename)

    v *= v_scale

    if isnothing(Δt)
        Δt = Δt_l
    end

    if isnothing(printerval)
        printerval = printerval_l
    end

    egf! = @time make_egf(1.0, 1.0, r)

    open(filename, "a") do io
        do_md(io, n_steps, Δt, egf!, r, v; add_first=false, t0=t)
    end
end

function rand_v(n, vs)
    v = (rand(3, n) .- 0.5) * vs
    net_v = sum(eachcol(v)) / n

    for vc in eachcol(v)
        vc .-= net_v
    end

    v
end

############ ANALYSIS #############

function get_tVK(filename)
    ts = Float64[]
    Vs = Float64[]
    Ks = Float64[]
    n_atm = 0
    open(filename) do io
        lines = Iterators.Stateful(eachline(io))

        while !isempty(lines)
            n_atm = parse(Int, popfirst!(lines))
            l = popfirst!(lines)
            ls = split(l, "; ")
            t, V, K = [parse(Float64, split(ls[i], " = ")[2]) for i in (2, 4, 5)]
            push!(ts, t)
            push!(Vs, V)
            push!(Ks, K)

            for _ in 1:n_atm
                popfirst!(lines)
            end
        end
    end
    ts, Vs, Ks, n_atm
end

function plot_tVK(filename; is=:, time=false, E0_i=0)
    ts, Vs, Ks = get_tVK(filename)

    if E0_i == 0
        E0_i = length(ts)
    end

    E0 = Vs[E0_i] + Ks[E0_i]
    @show E0
    Vs .-= E0

    if time
        plot(ts[is], Vs[is]; label="Potential", leg=:topleft)
        plot!(ts[is], Ks[is]; label="Kinetic")
        plot!(ts[is], (Vs+Ks)[is]; label="Total")
    else
        plot(Vs[is]; label="Potential", leg=:topleft)
        plot!(Ks[is]; label="Kinetic")
        plot!((Vs+Ks)[is]; label="Total")
    end
end

############ TESTS ################

function test_md()
    r = setup_cubic_lattice(20, 1.0)

    egf! = make_egf(1.0, 1.0, r)

    open("xyz/cube.xyz", "w") do io
        do_md(io, 1000, 0.01, egf!, r; printerval=10)
    end
end

function init_cold()
    r = evalfile("cold2.txt")

    egf! = make_egf(1.0, 1.0, r)

    open("xyz/20.xyz", "w") do io
        do_md(io, 1000, 0.01, egf!, r; printerval=10)
    end
end
