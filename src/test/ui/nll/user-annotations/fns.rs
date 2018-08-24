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

fn some_fn<T>(arg: T) { }

fn no_annot() {
    let c = 66;
    some_fn(&c); // OK
}

fn annot_underscore() {
    let c = 66;
    some_fn::<_>(&c); // OK
}

fn annot_reference_any_lifetime() {
    let c = 66;
    some_fn::<&u32>(&c); // OK
}

fn annot_reference_static_lifetime() {
    let c = 66;
    some_fn::<&'static u32>(&c); //~ ERROR
}

fn annot_reference_named_lifetime<'a>(_d: &'a u32) {
    let c = 66;
    some_fn::<&'a u32>(&c); //~ ERROR
}

fn annot_reference_named_lifetime_ok<'a>(c: &'a u32) {
    some_fn::<&'a u32>(c);
}

fn annot_reference_named_lifetime_in_closure<'a>(_: &'a u32) {
    let _closure = || {
        let c = 66;
        some_fn::<&'a u32>(&c); //~ ERROR
    };
}

fn annot_reference_named_lifetime_in_closure_ok<'a>(c: &'a u32) {
    let _closure = || {
        some_fn::<&'a u32>(c);
    };
}

fn main() { }
