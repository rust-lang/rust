// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



pub fn main() {
    let a = 1.5e6;
    let b = 1.5E6;
    let c = 1e6;
    let d = 1E6;
    let e = 3.0f32;
    let f = 5.9f64;
    let g = 1e6f32;
    let h = 1.0e7f64;
    let i = 1.0E7f64;
    let j = 3.1e+9;
    let k = 3.2e-10;
    fail_unless!((a == b));
    fail_unless!((c < b));
    fail_unless!((c == d));
    fail_unless!((e < g));
    fail_unless!((f < h));
    fail_unless!((g == 1000000.0f32));
    fail_unless!((h == i));
    fail_unless!((j > k));
    fail_unless!((k < a));
}
