// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let caller<F> = |f: F|  //~ ERROR expected one of `:`, `;`, `=`, or `@`, found `<`
    where F: Fn() -> i32
    {
        let x = f();
        println!("Y {}",x);
        return x;
    };

    caller(bar_handler);
}

fn bar_handler() -> i32 {
    5
}
