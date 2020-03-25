@enum Diffe begin
   Duplicate = 1
   Output = 2
   Constant = 3
end

function whatType(@nospecialize(dt))
    if <:(dt, Array)
        sub = whatType(eltype(dt))
        if sub == "diffe_dup"
            return "diffe_dup"
        elseif sub == "diffe_out"
            return "diffe_dup"
        else
            @assert(sub == "diffe_const")
            return "diffe_const"
        end
    end
    if <:(dt, Real)
        return "diffe_out"
    end
    if <:(dt, Int)
        return "diffe_const"
    end
    if <:(dt, String)
        return "diffe_const"
    end

    if !hasfieldcount(dt)
        # just be safe for now
        return "diffe_dup"
    end

    @assert(hasfieldcount(dt))
    @assert(isstructtype(dt))
    passpointer = true
    if passpointer
        ty = "diffe_const"
        for (ft, fn) in zip(fieldtypes(dt), fieldnames(dt))
            sub = whatType(ft)
            if sub == "diffe_dup"
                ty = "diffe_dup"
            elseif sub == "diffe_out"
                ty = "diffe_dup"
            else
                @assert(sub == "diffe_const")
            end
        end
        return ty
    else
        ty = "diffe_const"
        for (ft, fn) in zip(fieldtypes(dt), fieldnames(dt))
            sub = whatType(ft)
            if sub == "diffe_dup"
                ty = "diffe_dup"
            elseif sub == "diffe_out"
                if ty != "diffe_dup"
                    ty = "diffe_out"
                end
            else
                @assert(sub == "diffe_const")
            end
        end
        return ty
    end
end