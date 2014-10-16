// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn inty(fun: proc(int) -> int) -> int {
    fun(100)
}

fn booly(fun: proc(bool) -> bool) -> bool {
    fun(true)
}

// Check usage and precedence of block arguments in expressions:
pub fn main() {
    let v = vec!(-1.0f64, 0.0, 1.0, 2.0, 3.0);

    // Statement form does not require parentheses:
    for i in v.iter() {
        println!("{}", *i);
    }

}
