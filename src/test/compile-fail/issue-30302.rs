// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Stack<T> {
    Nil,
    Cons(T, Box<Stack<T>>)
}

fn is_empty<T>(s: Stack<T>) -> bool {
    match s {
        Nil => true,
//~^ WARN pattern binding `Nil` is named the same as one of the variants of the type `Stack`
//~| HELP consider making the path in the pattern qualified: `Stack::Nil`
        _ => false
//~^ ERROR unreachable pattern
    }
}

fn main() {}
