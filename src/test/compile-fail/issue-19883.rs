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
    type Output;

    fn from(src: Src) -> <Self as From<Src>>::Output;
}

trait To {
    // This is a typo, the return type should be `<Dst as From<Self>>::Output`
    fn to<Dst: From<Self>>(
        self
        //~^ error: the trait `core::marker::Sized` is not implemented
    ) ->
        <Dst as From<Self>>::Dst
        //~^ error: the trait `core::marker::Sized` is not implemented
    {
        From::from(
            //~^ error: the trait `core::marker::Sized` is not implemented
            self
        )
    }
}

fn main() {}
