// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Tr<T> {
    fn op(T) -> Self;
}

// these compile as if Self: Tr<U>, even tho only Self: Tr<Self or T>
trait A:    Tr<Self> {
    fn test<U>(u: U) -> Self {
        Tr::op(u)   //~ ERROR type mismatch
    }
}
trait B<T>: Tr<T> {
    fn test<U>(u: U) -> Self {
        Tr::op(u)   //~ ERROR type mismatch
    }
}

impl<T> Tr<T> for T {
    fn op(t: T) -> T { t }
}
impl<T> A for T {}

fn main() {
    std::io::println(A::test((&7306634593706211700, 8)));
}

