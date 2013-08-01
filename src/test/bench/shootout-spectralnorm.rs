// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::from_str::FromStr;
use std::os;
use std::vec;

#[inline]
fn A(i: i32, j: i32) -> i32 {
    (i+j) * (i+j+1) / 2 + i + 1
}

fn dot(v: &[f64], u: &[f64]) -> f64 {
    let mut sum = 0.0;
    foreach (i, &v_i) in v.iter().enumerate() {
        sum += v_i * u[i];
    }
    sum
}

fn mult_Av(v: &mut [f64], out: &mut [f64]) {
    foreach (i, out_i) in out.mut_iter().enumerate() {
        let mut sum = 0.0;
        foreach (j, &v_j) in v.mut_iter().enumerate() {
            sum += v_j / (A(i as i32, j as i32) as f64);
        }
        *out_i = sum;
    }
}

fn mult_Atv(v: &mut [f64], out: &mut [f64]) {
    foreach (i, out_i) in out.mut_iter().enumerate() {
        let mut sum = 0.0;
        foreach (j, &v_j) in v.mut_iter().enumerate() {
            sum += v_j / (A(j as i32, i as i32) as f64);
        }
        *out_i = sum;
    }
}

fn mult_AtAv(v: &mut [f64], out: &mut [f64], tmp: &mut [f64]) {
    mult_Av(v, tmp);
    mult_Atv(tmp, out);
}

#[fixed_stack_segment]
fn main() {
    let n: uint = FromStr::from_str(os::args()[1]).get();
    let mut u = vec::from_elem(n, 1f64);
    let mut v = u.clone();
    let mut tmp = u.clone();
    do 8.times {
        mult_AtAv(u, v, tmp);
        mult_AtAv(v, u, tmp);
    }

    printfln!("%.9f", (dot(u,v) / dot(v,v)).sqrt() as float);
}
