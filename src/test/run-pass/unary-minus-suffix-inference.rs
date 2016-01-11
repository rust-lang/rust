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
    let a = 1;
    let a_neg: i8 = -a;
    println!("{}", a_neg);

    let b = 1;
    let b_neg: i16 = -b;
    println!("{}", b_neg);

    let c = 1;
    let c_neg: i32 = -c;
    println!("{}", c_neg);

    let d = 1;
    let d_neg: i64 = -d;
    println!("{}", d_neg);

    let e = 1;
    let e_neg: isize = -e;
    println!("{}", e_neg);
}
