// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(PartialEq)]
struct Bar;
struct Baz;
struct Foo;
struct Fu;

impl PartialEq for Baz { fn eq(&self, _: &Baz) -> bool  { true } }

impl PartialEq<Fu> for Foo { fn eq(&self, _: &Fu) -> bool { true } }
impl PartialEq<Foo> for Fu { fn eq(&self, _: &Foo) -> bool { true } }

impl PartialEq<Bar> for Foo { fn eq(&self, _: &Bar) -> bool { false } }
impl PartialEq<Foo> for Bar { fn eq(&self, _: &Foo) -> bool { false } }

fn main() {
    assert!(Bar != Foo);
    assert!(Foo != Bar);

    assert!(Bar == Bar);

    assert!(Baz == Baz);

    assert!(Foo == Fu);
    assert!(Fu == Foo);
}
