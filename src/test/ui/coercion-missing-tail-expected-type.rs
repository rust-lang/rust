// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// #41425 -- error message "mismatched types" has wrong types

fn plus_one(x: i32) -> i32 { //~ ERROR mismatched types
    x + 1;
}

fn foo() -> Result<u8, u64> { //~ ERROR mismatched types
    Ok(1);
}

fn main() {
    let x = plus_one(5);
    println!("X = {}", x);
}
