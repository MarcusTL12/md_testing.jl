using OhMyREPL
using ForwardDiff
using ReverseDiff
using Optim

include("lennard-jones.jl")

function print_xyz(io, r; scale=2.0)
    println(io, size(r, 2))
    println(io)
    for c in eachcol(r)
        println(io, "H    ",
            c[1] * scale, " ",
            c[2] * scale, " ",
            c[3] * scale)
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

function geoopt(pot, r)
    @time g! = make_rd_grad(pot, r)
    @time o = optimize(pot, g!, r)
end
