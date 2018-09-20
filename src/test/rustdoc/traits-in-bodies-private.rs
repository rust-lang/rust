// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// when implementing the fix for traits-in-bodies, there was an ICE when documenting private items
// and a trait was defined in non-module scope

// compile-flags:--document-private-items

// @has traits_in_bodies_private/struct.SomeStruct.html
// @!has - '//code' 'impl HiddenTrait for SomeStruct'
pub struct SomeStruct;

fn __implementation_details() {
    trait HiddenTrait {}
    impl HiddenTrait for SomeStruct {}
}
