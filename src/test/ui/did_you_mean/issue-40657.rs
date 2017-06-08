// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn f1() { println!("1"); }
fn f2() { println!("2"); }

struct S(fn());
const CONST: S = S(f1);

fn main() {
    let x = S(f2);

    match x {
        CONST => {
            println!("match");
        }
        _ => {}
    }
}
