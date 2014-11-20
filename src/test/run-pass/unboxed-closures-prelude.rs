// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that the reexports of `FnOnce` et al from the prelude work.

#![feature(unboxed_closures)]

fn main() {
    let task: Box<Fn(int) -> int> = box |&: x| x;
    task.call((0i, ));

    let mut task: Box<FnMut(int) -> int> = box |&mut: x| x;
    task.call_mut((0i, ));

    call(|:x| x, 22);
}

fn call<F:FnOnce(int) -> int>(f: F, x: int) -> int {
    f.call_once((x,))
}

