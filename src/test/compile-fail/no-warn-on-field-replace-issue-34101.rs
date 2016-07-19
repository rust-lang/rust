// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue 34101: Circa 2016-06-05, `fn inline` below issued an
// erroneous warning from the elaborate_drops pass about moving out of
// a field in `Foo`, which has a destructor (and thus cannot have
// content moved out of it). The reason that the warning is erroneous
// in this case is that we are doing a *replace*, not a move, of the
// content in question, and it is okay to replace fields within `Foo`.
//
// Another more subtle problem was that the elaborate_drops was
// creating a separate drop flag for that internally replaced content,
// even though the compiler should enforce an invariant that any drop
// flag for such subcontent of `Foo` will always have the same value
// as the drop flag for `Foo` itself.
//
// This test is structured in a funny way; we cannot test for emission
// of the warning in question via the lint system, and therefore
// `#![deny(warnings)]` does nothing to detect it.
//
// So instead we use `#[rustc_error]` and put the test into
// `compile_fail`, where the emitted warning *will* be caught.

#![feature(rustc_attrs)]

struct Foo(String);

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn inline() {
    // (dummy variable so `f` gets assigned `var1` in MIR for both fn's)
    let _s = ();
    let mut f = Foo(String::from("foo"));
    f.0 = String::from("bar");
}

fn outline() {
    let _s = String::from("foo");
    let mut f = Foo(_s);
    f.0 = String::from("bar");
}

#[rustc_error]
fn main() { //~ ERROR compilation successful
    inline();
    outline();
}
