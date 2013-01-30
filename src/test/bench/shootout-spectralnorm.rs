// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Based on spectalnorm.gcc by Sebastien Loisel

extern mod std;

fn eval_A(i: uint, j: uint) -> float {
    1.0/(((i+j)*(i+j+1u)/2u+i+1u) as float)
}

fn eval_A_times_u(u: &[const float], Au: &mut [float]) {
    let N = vec::len(u);
    let mut i = 0u;
    while i < N {
        Au[i] = 0.0;
        let mut j = 0u;
        while j < N {
            Au[i] += eval_A(i, j) * u[j];
            j += 1u;
        }
        i += 1u;
    }
}

fn eval_At_times_u(u: &[const float], Au: &mut [float]) {
    let N = vec::len(u);
    let mut i = 0u;
    while i < N {
        Au[i] = 0.0;
        let mut j = 0u;
        while j < N {
            Au[i] += eval_A(j, i) * u[j];
            j += 1u;
        }
        i += 1u;
    }
}

fn eval_AtA_times_u(u: &[const float], AtAu: &mut [float]) {
    let v = vec::cast_to_mut(vec::from_elem(vec::len(u), 0.0));
    eval_A_times_u(u, v);
    eval_At_times_u(v, AtAu);
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"2000"]
    } else if args.len() <= 1u {
        ~[~"", ~"1000"]
    } else {
        args
    };

    let N = uint::from_str(args[1]).get();

    let u = vec::cast_to_mut(vec::from_elem(N, 1.0));
    let v = vec::cast_to_mut(vec::from_elem(N, 0.0));
    let mut i = 0u;
    while i < 10u {
        eval_AtA_times_u(u, v);
        eval_AtA_times_u(v, u);
        i += 1u;
    }

    let mut vBv = 0.0;
    let mut vv = 0.0;
    let mut i = 0u;
    while i < N {
        vBv += u[i] * v[i];
        vv += v[i] * v[i];
        i += 1u;
    }

    io::println(fmt!("%0.9f\n", float::sqrt(vBv / vv)));
}
