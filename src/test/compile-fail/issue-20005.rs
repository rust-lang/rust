// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait From<Src> {
    type Result;

    fn from(src: Src) -> Self::Result;
}

trait To {
    fn to<Dst>(  //~ ERROR the trait `core::marker::Sized` is not implemented
        self
    ) -> <Dst as From<Self>>::Result where Dst: From<Self> {
        From::from(self)
    }
}

fn main() {}
