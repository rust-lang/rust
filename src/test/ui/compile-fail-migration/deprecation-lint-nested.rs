// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(deprecated)]
#![allow(warnings)]

#[deprecated]
fn issue_35128() {
    format_args!("foo");
}

#[deprecated]
fn issue_35128_minimal() {
    static FOO: &'static str = "foo";
    let _ = FOO;
}

#[deprecated]
mod silent {
    type DeprecatedType = u8;
    struct DeprecatedStruct;
    fn deprecated_fn() {}
    trait DeprecatedTrait {}
    static DEPRECATED_STATIC: u8 = 0;
    const DEPRECATED_CONST: u8 = 1;

    struct Foo(DeprecatedType);

    impl DeprecatedTrait for Foo {}

    impl Foo {
        fn bar<T: DeprecatedTrait>() {
            deprecated_fn();
        }
    }

    fn foo() -> u8 {
        DEPRECATED_STATIC +
        DEPRECATED_CONST
    }
}

#[deprecated]
mod loud {
    #[deprecated]
    type DeprecatedType = u8;
    #[deprecated]
    struct DeprecatedStruct;
    #[deprecated]
    fn deprecated_fn() {}
    #[deprecated]
    trait DeprecatedTrait {}
    #[deprecated]
    static DEPRECATED_STATIC: u8 = 0;
    #[deprecated]
    const DEPRECATED_CONST: u8 = 1;

    struct Foo(DeprecatedType); //~ ERROR use of deprecated item

    impl DeprecatedTrait for Foo {} //~ ERROR use of deprecated item

    impl Foo {
        fn bar<T: DeprecatedTrait>() { //~ ERROR use of deprecated item
            deprecated_fn(); //~ ERROR use of deprecated item
        }
    }

    fn foo() -> u8 {
        DEPRECATED_STATIC + //~ ERROR use of deprecated item
        DEPRECATED_CONST //~ ERROR use of deprecated item
    }
}

fn main() {}
