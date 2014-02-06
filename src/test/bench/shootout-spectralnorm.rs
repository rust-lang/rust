// Copyright 2012-2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test arcs no longer unwrap

extern mod sync;

use std::from_str::FromStr;
use std::iter::count;
use std::cmp::min;
use std::os;
use std::vec::from_elem;
use sync::Arc;
use sync::RWArc;

fn A(i: uint, j: uint) -> f64 {
    ((i + j) * (i + j + 1) / 2 + i + 1) as f64
}

fn dot(v: &[f64], u: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (&v_i, &u_i) in v.iter().zip(u.iter()) {
        sum += v_i * u_i;
    }
    sum
}

fn mult(v: RWArc<~[f64]>, out: RWArc<~[f64]>, f: fn(&~[f64], uint) -> f64) {
    let wait = Arc::new(());
    let len = out.read(|out| out.len());
    let chunk = len / 100 + 1;
    for chk in count(0, chunk) {
        if chk >= len {break;}
        let w = wait.clone();
        let v = v.clone();
        let out = out.clone();
        spawn(proc() {
            for i in range(chk, min(len, chk + chunk)) {
                let val = v.read(|v| f(v, i));
                out.write(|out| out[i] = val);
            }
            let _ = w;
        });
    }
    let _ = wait.unwrap();
}

fn mult_Av_impl(v: &~[f64], i: uint) -> f64 {
    let mut sum = 0.;
    for (j, &v_j) in v.iter().enumerate() {
        sum += v_j / A(i, j);
    }
    sum
}

fn mult_Av(v: RWArc<~[f64]>, out: RWArc<~[f64]>) {
    mult(v, out, mult_Av_impl);
}

fn mult_Atv_impl(v: &~[f64], i: uint) -> f64 {
    let mut sum = 0.;
    for (j, &v_j) in v.iter().enumerate() {
        sum += v_j / A(j, i);
    }
    sum
}

fn mult_Atv(v: RWArc<~[f64]>, out: RWArc<~[f64]>) {
    mult(v, out, mult_Atv_impl);
}

fn mult_AtAv(v: RWArc<~[f64]>, out: RWArc<~[f64]>, tmp: RWArc<~[f64]>) {
    mult_Av(v, tmp.clone());
    mult_Atv(tmp, out);
}

fn main() {
    let args = os::args();
    let n = if os::getenv("RUST_BENCH").is_some() {
        5500
    } else if args.len() < 2 {
        2000
    } else {
        FromStr::from_str(args[1]).unwrap()
    };
    let u = RWArc::new(from_elem(n, 1.));
    let v = RWArc::new(from_elem(n, 1.));
    let tmp = RWArc::new(from_elem(n, 1.));
    for _ in range(0, 10) {
        mult_AtAv(u.clone(), v.clone(), tmp.clone());
        mult_AtAv(v.clone(), u.clone(), tmp.clone());
    }
    let u = u.unwrap();
    let v = v.unwrap();
    println!("{:.9f}", (dot(u,v) / dot(v,v)).sqrt());
}
