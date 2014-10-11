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

#![feature(macro_rules)]
#![feature(simd)]
#![allow(experimental)]

// ignore-pretty very bad with line comments

use std::io;
use std::os;
use std::simd::f64x2;
use std::sync::{Arc, Future};

const ITER: int = 50;
const LIMIT: f64 = 2.0;
const WORKERS: uint = 16;

#[inline(always)]
fn mandelbrot<W: io::Writer>(w: uint, mut out: W) -> io::IoResult<()> {
    assert!(WORKERS % 2 == 0);

    // Ensure w and h are multiples of 8.
    let w = (w + 7) / 8 * 8;
    let h = w;

    let chunk_size = h / WORKERS;

    // Account for remainders in workload division, e.g. 1000 / 16 = 62.5
    let last_chunk_size = if h % WORKERS != 0 {
        chunk_size + h % WORKERS
    } else {
        chunk_size
    };

    // precalc values
    let inverse_w_doubled = 2.0 / w as f64;
    let inverse_h_doubled = 2.0 / h as f64;
    let v_inverses = f64x2(inverse_w_doubled, inverse_h_doubled);
    let v_consts = f64x2(1.5, 1.0);

    // A lot of this code assumes this (so do other lang benchmarks)
    assert!(w == h);
    let mut precalc_r = Vec::with_capacity(w);
    let mut precalc_i = Vec::with_capacity(h);

    let precalc_futures = Vec::from_fn(WORKERS, |i| {
        Future::spawn(proc () {
            let mut rs = Vec::with_capacity(w / WORKERS);
            let mut is = Vec::with_capacity(w / WORKERS);

            let start = i * chunk_size;
            let end = if i == (WORKERS - 1) {
                start + last_chunk_size
            } else {
                (i + 1) * chunk_size
            };

            // This assumes w == h
            for x in range(start, end) {
                let xf = x as f64;
                let xy = f64x2(xf, xf);

                let f64x2(r, i) = xy * v_inverses - v_consts;
                rs.push(r);
                is.push(i);
            }

            (rs, is)
        })
    });

    for res in precalc_futures.into_iter() {
        let (rs, is) = res.unwrap();
        precalc_r.extend(rs.into_iter());
        precalc_i.extend(is.into_iter());
    }

    assert_eq!(precalc_r.len(), w);
    assert_eq!(precalc_i.len(), h);

    let arc_init_r = Arc::new(precalc_r);
    let arc_init_i = Arc::new(precalc_i);

    let data = Vec::from_fn(WORKERS, |i| {
        let vec_init_r = arc_init_r.clone();
        let vec_init_i = arc_init_i.clone();

        Future::spawn(proc () {
            let mut res: Vec<u8> = Vec::with_capacity((chunk_size * w) / 8);
            let init_r_slice = vec_init_r.as_slice();

            let start = i * chunk_size;
            let end = if i == (WORKERS - 1) {
                start + last_chunk_size
            } else {
                (i + 1) * chunk_size
            };

            for &init_i in vec_init_i.slice(start, end).iter() {
                write_line(init_i, init_r_slice, &mut res);
            }

            res
        })
    });

    try!(writeln!(&mut out as &mut Writer, "P4\n{} {}", w, h));
    for res in data.into_iter() {
        try!(out.write(res.unwrap().as_slice()));
    }
    out.flush()
}

fn write_line(init_i: f64, vec_init_r: &[f64], res: &mut Vec<u8>) {
    let v_init_i : f64x2 = f64x2(init_i, init_i);
    let v_2 : f64x2 = f64x2(2.0, 2.0);
    const LIMIT_SQUARED: f64 = LIMIT * LIMIT;

    for chunk_init_r in vec_init_r.chunks(8) {
        let mut cur_byte = 0xff;
        let mut i = 0;

        while i < 8 {
            let v_init_r = f64x2(chunk_init_r[i], chunk_init_r[i + 1]);
            let mut cur_r = v_init_r;
            let mut cur_i = v_init_i;
            let mut r_sq = v_init_r * v_init_r;
            let mut i_sq = v_init_i * v_init_i;

            let mut b = 0;
            for _ in range(0, ITER) {
                let r = cur_r;
                let i = cur_i;

                cur_i = v_2 * r * i + v_init_i;
                cur_r = r_sq - i_sq + v_init_r;

                let f64x2(bit1, bit2) = r_sq + i_sq;

                if bit1 > LIMIT_SQUARED {
                    b |= 2;
                    if b == 3 { break; }
                }

                if bit2 > LIMIT_SQUARED {
                    b |= 1;
                    if b == 3 { break; }
                }

                r_sq = cur_r * cur_r;
                i_sq = cur_i * cur_i;
            }

            cur_byte = (cur_byte << 2) + b;
            i += 2;
        }

        res.push(cur_byte^-1);
    }
}

fn main() {
    let args = os::args();
    let args = args.as_slice();
    let res = if args.len() < 2 {
        println!("Test mode: do not dump the image because it's not utf8, \
                  which interferes with the test runner.");
        mandelbrot(1000, io::util::NullWriter)
    } else {
        mandelbrot(from_str(args[1].as_slice()).unwrap(), io::stdout())
    };
    res.unwrap();
}
