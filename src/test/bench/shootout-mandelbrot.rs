// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate sync;

use std::io;
use sync::Future;

static ITER: int = 50;
static LIMIT: f64 = 2.0;

fn write_line(init_i: f64, vec_init_r: &[f64], res: &mut Vec<u8>) {
    for chunk_init_r in vec_init_r.chunks(8) {
        let mut cur_byte = 0xff;
        let mut cur_bitmask = 0x80;
        for &init_r in chunk_init_r.iter() {
            let mut cur_r = init_r;
            let mut cur_i = init_i;
            for _ in range(0, ITER) {
                let r = cur_r;
                let i = cur_i;
                cur_r = r * r - i * i + init_r;
                cur_i = 2.0 * r * i + init_i;

                if r * r + i * i > LIMIT * LIMIT {
                    cur_byte &= !cur_bitmask;
                    break;
                }
            }
            cur_bitmask >>= 1;
        }
        res.push(cur_byte);
    }
}

fn mandelbrot<W: io::Writer>(w: uint, mut out: W) -> io::IoResult<()> {
    // Ensure w and h are multiples of 8.
    let w = (w + 7) / 8 * 8;
    let h = w;
    let chunk_size = h / 8;

    let data: Vec<Future<Vec<u8>>> = range(0u, 8).map(|i| Future::spawn(proc () {
        let vec_init_r = Vec::from_fn(w, |x| 2.0 * (x as f64) / (w as f64) - 1.5);
        let mut res: Vec<u8> = Vec::with_capacity((chunk_size * w) / 8);
        for y in range(i * chunk_size, (i + 1) * chunk_size) {
            let init_i = 2.0 * (y as f64) / (h as f64) - 1.0;
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
