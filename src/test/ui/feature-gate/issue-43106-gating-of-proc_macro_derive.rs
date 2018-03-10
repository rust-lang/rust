// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// At time of authorship, #[proc_macro_derive = "2500"] will emit an
// error when it occurs on a mod (apart from crate-level), but will
// not descend further into the mod for other occurrences of the same
// error.
//
// This file sits on its own because the the "weird" occurrences here
// signal errors, making it incompatible with the "warnings only"
// nature of issue-43106-gating-of-builtin-attrs.rs

#[proc_macro_derive = "2500"]
//~^ ERROR the `#[proc_macro_derive]` attribute may only be used on bare functions
mod proc_macro_derive1 {
    mod inner { #![proc_macro_derive="2500"] }
    // (no error issued here if there was one on outer module)
}

mod proc_macro_derive2 {
    mod inner { #![proc_macro_derive="2500"] }
    //~^ ERROR the `#[proc_macro_derive]` attribute may only be used on bare functions

    #[proc_macro_derive = "2500"] fn f() { }
    //~^ ERROR the `#[proc_macro_derive]` attribute is only usable with crates of the `proc-macro`

    #[proc_macro_derive = "2500"] struct S;
    //~^ ERROR the `#[proc_macro_derive]` attribute may only be used on bare functions

    #[proc_macro_derive = "2500"] type T = S;
    //~^ ERROR the `#[proc_macro_derive]` attribute may only be used on bare functions

    #[proc_macro_derive = "2500"] impl S { }
    //~^ ERROR the `#[proc_macro_derive]` attribute may only be used on bare functions
}
