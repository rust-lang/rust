// Based on spectalnorm.gcc by Sebastien Loisel

use std;

fn eval_A(i: uint, j: uint) -> float {
    1.0/(((i+j)*(i+j+1u)/2u+i+1u) as float)
}

fn eval_A_times_u(u: [const float], Au: [mutable float]) {
    let N = vec::len(u);
    let i = 0u;
    while i < N {
        Au[i] = 0.0;
        let j = 0u;
        while j < N {
            Au[i] += eval_A(i, j) * u[j];
            j += 1u;
        }
        i += 1u;
    }
}

fn eval_At_times_u(u: [const float], Au: [mutable float]) {
    let N = vec::len(u);
    let i = 0u;
    while i < N {
        Au[i] = 0.0;
        let j = 0u;
        while j < N {
            Au[i] += eval_A(j, i) * u[j];
            j += 1u;
        }
        i += 1u;
    }
}

fn eval_AtA_times_u(u: [const float], AtAu: [mutable float]) {
    let v = vec::to_mut(vec::from_elem(vec::len(u), 0.0));
    eval_A_times_u(u, v);
    eval_At_times_u(v, AtAu);
}

fn main(args: [str]) {

    let N = if vec::len(args) == 2u {
        option::get(uint::from_str(args[1]))
    } else {
        1000u
    };

    let u = vec::to_mut(vec::from_elem(N, 1.0));
    let v = vec::to_mut(vec::from_elem(N, 0.0));
    let i = 0u;
    while i < 10u {
        eval_AtA_times_u(u, v);
        eval_AtA_times_u(v, u);
        i += 1u;
    }

    let vBv = 0.0;
    let vv = 0.0;
    let i = 0u;
    while i < N {
        vBv += u[i] * v[i];
        vv += v[i] * v[i];
        i += 1u;
    }

    io::println(#fmt("%0.9f\n", float::sqrt(vBv / vv)));
}
