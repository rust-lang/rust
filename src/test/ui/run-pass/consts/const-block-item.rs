// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


mod foo {
    pub trait Value {
        fn value(&self) -> usize;
    }
}

static BLOCK_USE: usize = {
    use foo::Value;
    100
};

static BLOCK_STRUCT_DEF: usize = {
    struct Foo {
        a: usize
    }
    Foo{ a: 300 }.a
};

static BLOCK_FN_DEF: fn(usize) -> usize = {
    fn foo(a: usize) -> usize {
        a + 10
    }
    foo
};

static BLOCK_MACRO_RULES: usize = {
    macro_rules! baz {
        () => (412)
    }
    baz!()
};

pub fn main() {
    assert_eq!(BLOCK_USE, 100);
    assert_eq!(BLOCK_STRUCT_DEF, 300);
    assert_eq!(BLOCK_FN_DEF(390), 400);
    assert_eq!(BLOCK_MACRO_RULES, 412);
}
