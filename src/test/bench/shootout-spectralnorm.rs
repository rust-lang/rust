// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(phase)]
#![allow(non_snake_case_functions)]
#[phase(plugin)] extern crate green;

use std::from_str::FromStr;
use std::iter::count;
use std::cmp::min;
use std::os;
use std::sync::{Arc, RWLock};

green_start!(main)

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

fn mult(v: Arc<RWLock<Vec<f64>>>, out: Arc<RWLock<Vec<f64>>>,
        f: fn(&Vec<f64>, uint) -> f64) {
    // We lanch in different tasks the work to be done.  To finish
    // this fuction, we need to wait for the completion of every
    // tasks.  To do that, we give to each tasks a wait_chan that we
    // drop at the end of the work.  At the end of this function, we
    // wait until the channel hang up.
    let (tx, rx) = channel();

    let len = out.read().len();
    let chunk = len / 20 + 1;
    for chk in count(0, chunk) {
        if chk >= len {break;}
        let tx = tx.clone();
        let v = v.clone();
        let out = out.clone();
        spawn(proc() {
            for i in range(chk, min(len, chk + chunk)) {
                let val = f(&*v.read(), i);
                *out.write().get_mut(i) = val;
            }
            drop(tx)
        });
    }

    // wait until the channel hang up (every task finished)
    drop(tx);
    for () in rx.iter() {}
}

fn mult_Av_impl(v: &Vec<f64> , i: uint) -> f64 {
    let mut sum = 0.;
    for (j, &v_j) in v.iter().enumerate() {
        sum += v_j / A(i, j);
    }
    sum
}

fn mult_Av(v: Arc<RWLock<Vec<f64>>>, out: Arc<RWLock<Vec<f64>>>) {
    mult(v, out, mult_Av_impl);
}

fn mult_Atv_impl(v: &Vec<f64> , i: uint) -> f64 {
    let mut sum = 0.;
    for (j, &v_j) in v.iter().enumerate() {
        sum += v_j / A(j, i);
    }
    sum
}

fn mult_Atv(v: Arc<RWLock<Vec<f64>>>, out: Arc<RWLock<Vec<f64>>>) {
    mult(v, out, mult_Atv_impl);
}

fn mult_AtAv(v: Arc<RWLock<Vec<f64>>>, out: Arc<RWLock<Vec<f64>>>,
             tmp: Arc<RWLock<Vec<f64>>>) {
    mult_Av(v, tmp.clone());
    mult_Atv(tmp, out);
}

fn main() {
    let args = os::args();
    let args = args.as_slice();
    let n = if os::getenv("RUST_BENCH").is_some() {
        5500
    } else if args.len() < 2 {
        2000
    } else {
        FromStr::from_str(args[1].as_slice()).unwrap()
    };
    let u = Arc::new(RWLock::new(Vec::from_elem(n, 1f64)));
    let v = Arc::new(RWLock::new(Vec::from_elem(n, 1f64)));
    let tmp = Arc::new(RWLock::new(Vec::from_elem(n, 1f64)));
    for _ in range(0u8, 10) {
        mult_AtAv(u.clone(), v.clone(), tmp.clone());
        mult_AtAv(v.clone(), u.clone(), tmp.clone());
    }

    let u = u.read();
    let v = v.read();
    println!("{:.9f}", (dot(u.as_slice(), v.as_slice()) /
                        dot(v.as_slice(), v.as_slice())).sqrt());
}
