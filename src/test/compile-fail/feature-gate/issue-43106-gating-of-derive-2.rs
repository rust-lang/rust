// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// `#![derive]` is interpreted (and raises errors) when it occurs at
// contexts other than ADT definitions. This test checks cases where
// the derive-macro does not exist.

#![derive(x3300)]
//~^ ERROR cannot find derive macro `x3300` in this scope

#[derive(x3300)]
//~^ ERROR cannot find derive macro `x3300` in this scope
mod derive {
    mod inner { #![derive(x3300)] }
    //~^ ERROR cannot find derive macro `x3300` in this scope

    #[derive(x3300)]
    //~^ ERROR cannot find derive macro `x3300` in this scope
    fn derive() { }

    #[derive(x3300)]
    //~^ ERROR cannot find derive macro `x3300` in this scope
    union U { f: i32 }

    #[derive(x3300)]
    //~^ ERROR cannot find derive macro `x3300` in this scope
    enum E { }

    #[derive(x3300)]
    //~^ ERROR cannot find derive macro `x3300` in this scope
    struct S;

    #[derive(x3300)]
    //~^ ERROR cannot find derive macro `x3300` in this scope
    type T = S;

    #[derive(x3300)]
    //~^ ERROR cannot find derive macro `x3300` in this scope
    impl S { }
}
