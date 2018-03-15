// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing gating of `#[rustc_deprecated]` in "weird" places.
//
// This file sits on its own because these signal errors, making
// this test incompatible with the "warnings only" nature of
// issue-43106-gating-of-builtin-attrs.rs

#![rustc_deprecated           = "1500"]
//~^ ERROR stability attributes may not be used outside of the standard library

#[rustc_deprecated = "1500"]
//~^ ERROR stability attributes may not be used outside of the standard library
mod rustc_deprecated {
    mod inner { #![rustc_deprecated="1500"] }
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[rustc_deprecated = "1500"] fn f() { }
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[rustc_deprecated = "1500"] struct S;
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[rustc_deprecated = "1500"] type T = S;
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[rustc_deprecated = "1500"] impl S { }
    //~^ ERROR stability attributes may not be used outside of the standard library
}

fn main() {}
