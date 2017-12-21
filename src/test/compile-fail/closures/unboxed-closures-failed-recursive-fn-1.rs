// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Various unsuccessful attempts to put the unboxed closure kind
// inference into an awkward position that might require fixed point
// iteration (basically where inferring the kind of a closure `c`
// would require knowing the kind of `c`). I currently believe this is
// impossible.

fn a() {
    // This case of recursion wouldn't even require fixed-point
    // iteration, but it still doesn't work. The weird structure with
    // the `Option` is to avoid giving any useful hints about the `Fn`
    // kind via the expected type.
    let mut factorial: Option<Box<Fn(u32) -> u32>> = None;

    let f = |x: u32| -> u32 {
        let g = factorial.as_ref().unwrap();
        //~^ ERROR `factorial` does not live long enough
        if x == 0 {1} else {x * g(x-1)}
    };

    factorial = Some(Box::new(f));
}

fn b() {
    let mut factorial: Option<Box<Fn(u32) -> u32 + 'static>> = None;

    let f = |x: u32| -> u32 {
        //~^ ERROR closure may outlive the current function, but it borrows `factorial`
        let g = factorial.as_ref().unwrap();
        if x == 0 {1} else {x * g(x-1)}
    };

    factorial = Some(Box::new(f));
}

fn main() { }
