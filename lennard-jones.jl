
function lennard_jones(r_eq, E_b)
    σ = 2^(-1//6) * r_eq
    function potential(r)
        # Number of particles
        n_p = size(r, 2)
        E = zero(eltype(r))
        for i in 1:n_p, j in (i + 1):n_p
            ri = @view r[:, i]
            rj = @view r[:, j]
            r2 = sum((a - b)^2 for (a, b) in zip(ri, rj))
            sr2 = σ^2 / r2
            sr6 = sr2^3
            sr12 = sr6^2

            E += sr12 - sr6
        end
        4E_b * E
    end
end
