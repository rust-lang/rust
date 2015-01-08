// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can use method notation to call methods based on a
// projection bound from a trait. Issue #20469.

///////////////////////////////////////////////////////////////////////////

trait MakeString {
    fn make_string(&self) -> String;
}

impl MakeString for int {
    fn make_string(&self) -> String {
        format!("{}", *self)
    }
}

impl MakeString for uint {
    fn make_string(&self) -> String {
        format!("{}", *self)
    }
}

///////////////////////////////////////////////////////////////////////////

trait Foo {
    type F: MakeString;

    fn get(&self) -> &Self::F;
}

fn foo<F:Foo>(f: &F) -> String {
    f.get().make_string()
}

///////////////////////////////////////////////////////////////////////////

struct SomeStruct {
    field: int,
}

impl Foo for SomeStruct {
    type F = int;

    fn get(&self) -> &int {
        &self.field
    }
}

///////////////////////////////////////////////////////////////////////////

struct SomeOtherStruct {
    field: uint,
}

impl Foo for SomeOtherStruct {
    type F = uint;

    fn get(&self) -> &uint {
        &self.field
    }
}

fn main() {
    let x = SomeStruct { field: 22 };
    assert_eq!(foo(&x), format!("22"));

    let x = SomeOtherStruct { field: 44 };
    assert_eq!(foo(&x), format!("44"));
}
