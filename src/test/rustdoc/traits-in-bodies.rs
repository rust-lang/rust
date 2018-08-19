// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//prior to fixing `everybody_loops` to preserve items, rustdoc would crash on this file, as it
//didn't see that `SomeStruct` implemented `Clone`

//FIXME(misdreavus): whenever rustdoc shows traits impl'd inside bodies, make sure this test
//reflects that

pub struct Bounded<T: Clone>(T);

pub struct SomeStruct;

fn asdf() -> Bounded<SomeStruct> {
    impl Clone for SomeStruct {
        fn clone(&self) -> SomeStruct {
            SomeStruct
        }
    }

    Bounded(SomeStruct)
}
