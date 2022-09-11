using LinearAlgebra

function lennard_jones(r_eq, E_b)
    σ2 = 2^(-1 // 3) * r_eq^2
    function potential(r)
        n_p = size(r, 2)
        E = 0.0
        @inbounds for i in 1:n_p, j in (i+1):n_p
            ri = @view r[:, i]
            rj = @view r[:, j]
            r2 = sum((a - b)^2 for (a, b) in zip(ri, rj))
            sr2 = σ2 / r2
            sr6 = sr2^3
            sr12 = sr6^2

            E += sr12 - sr6
        end
        4E_b * E
    end
end

function lennard_jones_par(r_eq, E_b)
    σ2 = 2^(-1 // 3) * r_eq^2
    nth = Threads.nthreads()
    buffers = zeros(nth)
    function potential(r)
        n_p = size(r, 2)
        fill!(buffers, 0.0)
        @inbounds Threads.@threads for id in 1:nth
            E_loc = 0.0
            for i in id:nth:n_p, j in (i+1):n_p
                ri = @view r[:, i]
                rj = @view r[:, j]
                r2 = sum((a - b)^2 for (a, b) in zip(ri, rj))
                sr2 = σ2 / r2
                sr6 = sr2^3
                sr12 = sr6^2

                E_loc += sr12 - sr6
            end
            buffers[id] += E_loc
        end
        4E_b * sum(buffers)
    end
end

function lennard_jones_grad(r_eq, E_b)
    σ2 = 2^(-1 // 3) * r_eq^2
    function g!(g, r)
        n_p = size(r, 2)
        fill!(g, 0)
        @inbounds for i in 1:n_p, j in (i+1):n_p
            ri = @view r[:, i]
            rj = @view r[:, j]
            r2 = sum((a - b)^2 for (a, b) in zip(ri, rj))
            a = σ2 / r2
            a3 = a^3
            s = -24E_b * a3 / r2 * (2a3 - 1)
            for q in 1:3
                gq = (rj[q] - ri[q]) * s
                g[q, i] -= gq
                g[q, j] += gq
            end
        end
        g
    end
end

function lennard_jones_grad_par(r_eq, E_b, r)
    σ2 = 2^(-1 // 3) * r_eq^2
    nth = Threads.nthreads()
    buffers = [similar(r) for _ in 1:nth]
    function g!(g, r)
        n_p = size(r, 2)
        @inbounds Threads.@threads for id in 1:nth
            buf = buffers[id]
            fill!(buf, 0.0)
            for i in id:nth:n_p, j in (i+1):n_p
                ri = @view r[:, i]
                rj = @view r[:, j]
                r2 = sum((a - b)^2 for (a, b) in zip(ri, rj))
                a = σ2 / r2
                a3 = a^3
                s = -24E_b * a3 / r2 * (2a3 - 1)
                for q in 1:3
                    gq = (rj[q] - ri[q]) * s
                    buf[q, i] -= gq
                    buf[q, j] += gq
                end
            end
        end
        fill!(g, 0.0)
        for buf in buffers
            axpy!(1.0, buf, g)
        end
        g
    end
end
