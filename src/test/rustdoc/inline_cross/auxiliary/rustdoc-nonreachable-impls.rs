// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct Foo;

pub trait Woof {}
pub trait Bark {}

mod private {
    // should be shown
    impl ::Woof for ::Foo {}

    pub trait Bar {}
    pub struct Wibble;

    // these should not be shown
    impl Bar for ::Foo {}
    impl Bar for Wibble {}
    impl ::Bark for Wibble {}
    impl ::Woof for Wibble {}
}

#[doc(hidden)]
pub mod hidden {
    // should be shown
    impl ::Bark for ::Foo {}

    pub trait Qux {}
    pub struct Wobble;


    // these should only be shown if they're reexported correctly
    impl Qux for ::Foo {}
    impl Qux for Wobble {}
    impl ::Bark for Wobble {}
    impl ::Woof for Wobble {}
}
