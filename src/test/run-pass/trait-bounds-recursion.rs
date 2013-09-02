// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait I { fn i(&self) -> Self; }

trait A<T:I> {
    fn id(x:T) -> T { x.i() }
}

trait J<T> { fn j(&self) -> Self; }

trait B<T:J<T>> {
    fn id(x:T) -> T { x.j() }
}

trait C {
    fn id<T:J<T>>(x:T) -> T { x.j() }
}

fn main() { }
