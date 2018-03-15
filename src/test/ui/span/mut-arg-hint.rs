// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait B {
    fn foo(mut a: &String) {
        a.push_str("bar");
    }
}

pub fn foo<'a>(mut a: &'a String) {
    a.push_str("foo");
}

struct A {}

impl A {
    pub fn foo(mut a: &String) {
        a.push_str("foo");
    }
}

fn main() {
    foo(&"a".to_string());
    A::foo(&"a".to_string());
}
