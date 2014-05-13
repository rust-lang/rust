// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(macro_rules)]

// ignore-pretty very bad with line comments

extern crate sync;

use std::io;
use sync::Future;

static ITER: int = 50;
static LIMIT: f64 = 2.0;

macro_rules! core_loop(
    ($pow:expr ~ $mask:expr: $ctx:ident, $b:ident) => (
        {
            let r = $ctx.r;
            let i = $ctx.i;

            $ctx.r = r * r - i * i + $ctx.init_r;
            $ctx.i = 2.0 * r * i + $ctx.init_i;

            if r * r + i * i > LIMIT * LIMIT {
                $b |= $pow;
                if $b == $mask { break; }
            }
        }
    );
)

#[inline(always)]
fn write_line(init_i: f64, vec_init_r: &[f64], res: &mut Vec<u8>) {
    struct Context { r: f64, i: f64, init_i: f64, init_r: f64 }
    impl Context {
        #[inline(always)]
        fn new(i: f64, r: f64) -> Context {
            Context { r: r, i: i, init_r: r, init_i: i }
        }
    }

    let mut cur_byte;
    let mut i;
    let mut bit_1;
    let mut bit_2;
    let mut b;
    for chunk_init_r in vec_init_r.chunks(8) {
        cur_byte = 0xff;
        i = 0;

        while i < 8 {
            bit_1 = Context::new(init_i, chunk_init_r[i]);
            bit_2 = Context::new(init_i, chunk_init_r[i + 1]);

            b = 0;
            for _ in range(0, ITER) {
                core_loop!(2 ~ 3: bit_1, b);
                core_loop!(1 ~ 3: bit_2, b);
            }

            cur_byte = (cur_byte << 2) + b;
            i += 2;
        }
        res.push(cur_byte^-1);
    }
}

fn mandelbrot<W: io::Writer>(w: uint, mut out: W) -> io::IoResult<()> {
    // Ensure w and h are multiples of 8.
    let w = (w + 7) / 8 * 8;
    let h = w;
    let inverse_w_doubled = 2.0 / w as f64;
    let inverse_h_doubled = 2.0 / h as f64;
    let chunk_size = h / 16;

    let data: Vec<Future<Vec<u8>>> = range(0u, 16).map(|i| Future::spawn(proc () {
        let vec_init_r = Vec::from_fn(w, |x| (x as f64) * inverse_w_doubled - 1.5);
        let mut res: Vec<u8> = Vec::with_capacity((chunk_size * w) / 8);
        for y in range(i * chunk_size, (i + 1) * chunk_size) {
            let init_i = (y as f64) * inverse_h_doubled - 1.0;
            write_line(init_i, vec_init_r.as_slice(), &mut res);
        }
        res
    })).collect();

    try!(writeln!(&mut out as &mut Writer, "P4\n{} {}", w, h));
    for res in data.move_iter() {
        try!(out.write(res.unwrap().as_slice()));
    }
    out.flush()
}

fn main() {
    let args = std::os::args();
    let args = args.as_slice();
    let res = if args.len() < 2 {
        println!("Test mode: do not dump the image because it's not utf8, \
                  which interferes with the test runner.");
        mandelbrot(1000, std::io::util::NullWriter)
    } else {
        mandelbrot(from_str(args[1]).unwrap(), std::io::stdout())
    };
    res.unwrap();
}
