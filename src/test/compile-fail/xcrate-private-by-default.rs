// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:static_priv_by_default.rs

#[allow(unused_imports)];
#[no_std];

extern mod static_priv_by_default;

fn foo<T>() {}

#[start]
fn main(_: int, _: **u8, _: *u8) -> int {
    // Actual public items should be public
    static_priv_by_default::a;
    static_priv_by_default::b;
    static_priv_by_default::c;
    foo::<static_priv_by_default::d>();

    // publicly re-exported items should be available
    static_priv_by_default::bar::e;
    static_priv_by_default::bar::f;
    static_priv_by_default::bar::g;
    foo::<static_priv_by_default::bar::h>();

    // private items at the top should be inaccessible
    static_priv_by_default::i;
    //~^ ERROR: unresolved name
    static_priv_by_default::j;
    //~^ ERROR: unresolved name
    static_priv_by_default::k;
    //~^ ERROR: unresolved name
    foo::<static_priv_by_default::l>();
    //~^ ERROR: use of undeclared type name
    //~^^ ERROR: use of undeclared type name

    // public items in a private mod should be inaccessible
    static_priv_by_default::foo::a;
    //~^ ERROR: unresolved name
    static_priv_by_default::foo::b;
    //~^ ERROR: unresolved name
    static_priv_by_default::foo::c;
    //~^ ERROR: unresolved name
    foo::<static_priv_by_default::foo::d>();
    //~^ ERROR: use of undeclared type name
    //~^^ ERROR: use of undeclared type name

    3
}
