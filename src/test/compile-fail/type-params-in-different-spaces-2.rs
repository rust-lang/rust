// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test static calls to make sure that we align the Self and input
// type parameters on a trait correctly.

trait Tr<T> : Sized {
    fn op(_: T) -> Self;
}

trait A:    Tr<Self> {
    fn test<U>(u: U) -> Self {
        Tr::op(u)   //~ ERROR E0277
    }
}

trait B<T>: Tr<T> {
    fn test<U>(u: U) -> Self {
        Tr::op(u)   //~ ERROR E0277
    }
}

fn main() {
}
