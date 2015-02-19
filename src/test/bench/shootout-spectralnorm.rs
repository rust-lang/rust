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

#![allow(non_snake_case)]
#![feature(unboxed_closures)]

use std::iter::{repeat, AdditiveIterator};
use std::thread;
use std::mem;
use std::num::Float;
use std::os;
use std::env;
use std::raw::Repr;
use std::simd::f64x2;

fn main() {
    let mut args = env::args();
    let answer = spectralnorm(if env::var_os("RUST_BENCH").is_some() {
        5500
    } else if args.len() < 2 {
        2000
    } else {
        args.nth(1).unwrap().parse().unwrap()
    });
    println!("{:.9}", answer);
}

fn spectralnorm(n: uint) -> f64 {
    assert!(n % 2 == 0, "only even lengths are accepted");
    let mut u = repeat(1.0).take(n).collect::<Vec<_>>();
    let mut v = u.clone();
    let mut tmp = v.clone();
    for _ in 0..10 {
        mult_AtAv(&u, &mut v, &mut tmp);
        mult_AtAv(&v, &mut u, &mut tmp);
    }
    (dot(&u, &v) / dot(&v, &v)).sqrt()
}

fn mult_AtAv(v: &[f64], out: &mut [f64], tmp: &mut [f64]) {
    mult_Av(v, tmp);
    mult_Atv(tmp, out);
}

fn mult_Av(v: &[f64], out: &mut [f64]) {
    parallel(out, |start, out| mult(v, out, start, |i, j| A(i, j)));
}

fn mult_Atv(v: &[f64], out: &mut [f64]) {
    parallel(out, |start, out| mult(v, out, start, |i, j| A(j, i)));
}

fn mult<F>(v: &[f64], out: &mut [f64], start: uint, a: F)
           where F: Fn(uint, uint) -> f64 {
    for (i, slot) in out.iter_mut().enumerate().map(|(i, s)| (i + start, s)) {
        let mut sum = f64x2(0.0, 0.0);
        for (j, chunk) in v.chunks(2).enumerate().map(|(j, s)| (2 * j, s)) {
            let top = f64x2(chunk[0], chunk[1]);
            let bot = f64x2(a(i, j), a(i, j + 1));
            sum += top / bot;
        }
        let f64x2(a, b) = sum;
        *slot = a + b;
    }
}

fn A(i: uint, j: uint) -> f64 {
    ((i + j) * (i + j + 1) / 2 + i + 1) as f64
}

fn dot(v: &[f64], u: &[f64]) -> f64 {
    v.iter().zip(u.iter()).map(|(a, b)| *a * *b).sum()
}


// Executes a closure in parallel over the given mutable slice. The closure `f`
// is run in parallel and yielded the starting index within `v` as well as a
// sub-slice of `v`.
fn parallel<'a,T, F>(v: &mut [T], ref f: F)
                  where T: Send + Sync + 'a,
                        F: Fn(uint, &mut [T]) + Sync + 'a {
    let size = v.len() / os::num_cpus() + 1;
    v.chunks_mut(size).enumerate().map(|(i, chunk)| {
        thread::scoped(move|| {
            f(i * size, chunk)
        })
    }).collect::<Vec<_>>();
}
