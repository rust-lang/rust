// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:syntax-extension-with-field-names.rs
// ignore-stage1

#![feature(phase)]

#[phase(plugin)]
extern crate "syntax-extension-with-field-names" as extension;

trait FieldNames {
    fn field_names(&self) -> Vec<&'static str>;
    fn static_field_names(_: Option<Self>) -> Vec<&'static str>;
}

#[deriving_field_names]
struct Foo {
    x: int,
    #[name = "$y"]
    y: int,
}

fn main() {
    let foo = Foo {
        x: 1,
        y: 2,
    };

    assert_eq!(foo.field_names(), vec!("x", "$y"));
    assert_eq!(FieldNames::static_field_names(None::<Foo>), vec!("x", "$y"));
}
