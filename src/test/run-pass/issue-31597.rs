// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Make {
    type Out;

    fn make() -> Self::Out;
}

impl Make for () {
    type Out = ();

    fn make() -> Self::Out {}
}

// Also make sure we don't hit an ICE when the projection can't be known
fn f<T: Make>() -> <T as Make>::Out { loop {} }

// ...and that it works with a blanket impl
trait Tr {
    type Assoc;
}

impl<T: Make> Tr for T {
    type Assoc = ();
}

fn g<T: Make>() -> <T as Tr>::Assoc { }

fn main() {}
