// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #48071. This test used to ICE because -- in
// the leak-check -- it would pass since we knew that the return type
// was `'static`, and hence `'static: 'a` was legal even for a
// placeholder region, but in NLL land it would fail because we had
// rewritten `'static` to a region variable.
//
// compile-pass

#![allow(warnings)]
#![feature(dyn_trait)]
#![feature(nll)]

trait Foo {
    fn foo(&self) { }
}

impl Foo for () {
}

type MakeFooFn = for<'a> fn(&'a u8) -> Box<dyn Foo + 'a>;

fn make_foo(x: &u8) -> Box<dyn Foo + 'static> {
    Box::new(())
}

fn main() {
    let x: MakeFooFn = make_foo as MakeFooFn;
}
