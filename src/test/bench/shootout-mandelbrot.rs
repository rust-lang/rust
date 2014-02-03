// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io;
use std::io::BufferedWriter;

struct DummyWriter;
impl Writer for DummyWriter {
    fn write(&mut self, _: &[u8]) -> io::IoResult<()> { Ok(()) }
}

static ITER: int = 50;
static LIMIT: f64 = 2.0;

fn main() {
    let args = std::os::args();
    let (w, mut out) = if args.len() < 2 {
        println!("Test mode: do not dump the image because it's not utf8, \
                  which interferes with the test runner.");
        (1000, ~DummyWriter as ~Writer)
    } else {
        (from_str(args[1]).unwrap(),
         ~BufferedWriter::new(std::io::stdout()) as ~Writer)
    };
    let h = w;
    let mut byte_acc = 0u8;
    let mut bit_num = 0;

    writeln!(out, "P4\n{} {}", w, h);

    for y in range(0, h) {
        let y = y as f64;
        for x in range(0, w) {
            let mut z_r = 0f64;
            let mut z_i = 0f64;
            let mut t_r = 0f64;
            let mut t_i = 0f64;
            let c_r = 2.0 * (x as f64) / (w as f64) - 1.5;
            let c_i = 2.0 * (y as f64) / (h as f64) - 1.0;

            for _ in range(0, ITER) {
                if t_r + t_i > LIMIT * LIMIT {
                    break;
                }

                z_i = 2.0 * z_r * z_i + c_i;
                z_r = t_r - t_i + c_r;
                t_r = z_r * z_r;
                t_i = z_i * z_i;
            }

            byte_acc <<= 1;
            if t_r + t_i <= LIMIT * LIMIT {
                byte_acc |= 1;
            }

            bit_num += 1;

            if bit_num == 8 {
                out.write_u8(byte_acc);
                byte_acc = 0;
                bit_num = 0;
            } else if x == w - 1 {
                byte_acc <<= 8 - w % 8;
                out.write_u8(byte_acc);
                byte_acc = 0;
                bit_num = 0;
            }
        }
    }

    out.flush();
}
