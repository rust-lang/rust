// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const x : (i32,i32) = (0xfeedf00dd,0xca11ab1e);
const y : { x: (i64, i64),
            y: { a: float,
                 b: float } } = { x: (0xf0f0f0f0_f0f0f0f0,
                                      0xabababab_abababab),
                                  y: { a: 3.14159265358979323846,
                                       b: 2.7182818284590452354 } };

fn main() {
    let (p, _) = y.x;
    assert p == 0xf0f0f0f0_f0f0f0f0;
    io::println(fmt!("0x%x", p as uint));
}
