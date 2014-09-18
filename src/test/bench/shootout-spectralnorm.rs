// The Computer Language Benchmarks Game
// http://benchmarksgame.alioth.debian.org/
//
// contributed by the Rust Project Developers

// Copyright (c) 2012-2014 The Rust Project Developers
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in
//   the documentation and/or other materials provided with the
//   distribution.
//
// - Neither the name of "The Computer Language Benchmarks Game" nor
//   the name of "The Computer Language Shootout Benchmarks" nor the
//   names of its contributors may be used to endorse or promote
//   products derived from this software without specific prior
//   written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

// no-pretty-expanded FIXME #15189

#![feature(phase)]
#![allow(non_snake_case)]
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
    // We launch in different tasks the work to be done.  To finish
    // this function, we need to wait for the completion of every
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
