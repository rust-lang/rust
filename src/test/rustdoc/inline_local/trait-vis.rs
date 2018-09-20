// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait ThisTrait {}

mod asdf {
    use ThisTrait;

    pub struct SomeStruct;

    impl ThisTrait for SomeStruct {}

    trait PrivateTrait {}

    impl PrivateTrait for SomeStruct {}
}

// @has trait_vis/struct.SomeStruct.html
// @has - '//code' 'impl ThisTrait for SomeStruct'
// !@has - '//code' 'impl PrivateTrait for SomeStruct'
pub use asdf::SomeStruct;
