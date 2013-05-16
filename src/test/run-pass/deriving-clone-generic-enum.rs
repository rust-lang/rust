// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Clone, DeepClone)]
enum E<T,U> {
    A(T),
    B(T,U),
    C
}

fn main() {
    let _ = A::<int, int>(1i).clone();
    let _ = B(1i, 1.234).deep_clone();
}
