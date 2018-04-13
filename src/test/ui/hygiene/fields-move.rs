// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// issue #46314

#![feature(decl_macro)]

#[derive(Debug)]
struct NonCopy(String);

struct Foo {
    x: NonCopy,
}

macro copy_modern($foo: ident) {
   $foo.x
}

macro_rules! copy_legacy {
    ($foo: ident) => {
        $foo.x //~ ERROR use of moved value: `foo.x`
    }
}

fn assert_two_copies(a: NonCopy, b: NonCopy) {
   println!("Got two copies: {:?}, {:?}", a, b);
}

fn main() {
    let foo = Foo { x: NonCopy("foo".into()) };
    assert_two_copies(copy_modern!(foo), foo.x); //~ ERROR use of moved value: `foo.x`
    assert_two_copies(copy_legacy!(foo), foo.x); //~ ERROR use of moved value: `foo.x`
}
