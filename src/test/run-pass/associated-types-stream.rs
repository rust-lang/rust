// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test references to the trait `Stream` in the bounds for associated
// types defined on `Stream`. Issue #20551.

trait Stream {
    type Car;
    type Cdr: Stream;

    fn car(&self) -> Self::Car;
    fn cdr(self) -> Self::Cdr;
}

impl Stream for () {
    type Car = ();
    type Cdr = ();
    fn car(&self) -> () { () }
    fn cdr(self) -> () { self }
}

impl<T,U> Stream for (T, U)
    where T : Clone, U : Stream
{
    type Car = T;
    type Cdr = U;
    fn car(&self) -> T { self.0.clone() }
    fn cdr(self) -> U { self.1 }
}

fn main() {
    let p = (22, (44, (66, ())));
    assert_eq!(p.car(), 22);

    let p = p.cdr();
    assert_eq!(p.car(), 44);

    let p = p.cdr();
    assert_eq!(p.car(), 66);
}
