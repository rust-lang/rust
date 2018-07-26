// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is just a check-list of the cases where feeding arguments to
// `#[macro_use]` is rejected. (The cases where no error is emitted
// corresponds to cases where the attribute is currently unused, so we
// get that warning; see issue-43106-gating-of-builtin-attrs.rs

#![macro_use                  = "4900"] //~ ERROR arguments to macro_use are not allowed here

#[macro_use = "2700"]
//~^ ERROR arguments to macro_use are not allowed here
mod macro_escape {
    mod inner { #![macro_use="2700"] }
    //~^ ERROR arguments to macro_use are not allowed here

    #[macro_use = "2700"] fn f() { }

    #[macro_use = "2700"] struct S;

    #[macro_use = "2700"] type T = S;

    #[macro_use = "2700"] impl S { }
}

fn main() { }
