using OhMyREPL
using ForwardDiff
using ReverseDiff
using Enzyme

include("lennard-jones.jl")

function setup_cubic_lattice(n, r)
    coords = Float64[]

    is = 0:(n-1)
    for i in is, j in is, k in is
        append!(coords, (i, j, k) .* r)
    end

    reshape(coords, 3, n^3)
end

function make_fd_grad(pot)
    function g!(g, r)
        ForwardDiff.gradient!(g, pot, r)
    end
end

function make_rd_grad(pot, r)
    tape = ReverseDiff.GradientTape(pot, r)
    comp_tape = ReverseDiff.compile(tape)

    function g!(g, r)
        ReverseDiff.gradient!(g, comp_tape, r)
    end
end

function make_ez_grad(pot)
    function g!(g, r)
        Enzyme.gradient!(Reverse, g, pot, r)
    end
end
