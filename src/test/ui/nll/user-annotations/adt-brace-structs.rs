// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Unit test for the "user substitutions" that are annotated on each
// node.

#![feature(nll)]

struct SomeStruct<T> { t: T }

fn no_annot() {
    let c = 66;
    SomeStruct { t: &c };
}

fn annot_underscore() {
    let c = 66;
    SomeStruct::<_> { t: &c };
}

fn annot_reference_any_lifetime() {
    let c = 66;
    SomeStruct::<&u32> { t: &c };
}

fn annot_reference_static_lifetime() {
    let c = 66;
    SomeStruct::<&'static u32> { t: &c }; //~ ERROR
}

fn annot_reference_named_lifetime<'a>(_d: &'a u32) {
    let c = 66;
    SomeStruct::<&'a u32> { t: &c }; //~ ERROR
}

fn annot_reference_named_lifetime_ok<'a>(c: &'a u32) {
    SomeStruct::<&'a u32> { t: c };
}

fn annot_reference_named_lifetime_in_closure<'a>(_: &'a u32) {
    let _closure = || {
        let c = 66;
        SomeStruct::<&'a u32> { t: &c }; //~ ERROR
    };
}

fn annot_reference_named_lifetime_in_closure_ok<'a>(c: &'a u32) {
    let _closure = || {
        SomeStruct::<&'a u32> { t: c };
    };
}

fn main() { }
