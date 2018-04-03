// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

trait A { }
trait B { }
trait C { }

trait MyTrait {
    fn foo(&self) -> &'static str;
}

impl<T: A> MyTrait for T {
    default fn foo(&self) -> &'static str {
        "implements_A"
    }
}
impl<T: B> MyTrait for T {
    default fn foo(&self) -> &'static str {
        "implements_B"
    }
}

// This would be OK:
impl<T: A + B> MyTrait for T {
    default fn foo(&self) -> &'static str {
        "implements_A+B"
    }
}
// But what about this:
impl<T: A + B + C> MyTrait for T {
    fn foo(&self) -> &'static str {
        "implements_A+B+C"
    }
}

struct S_A;
struct S_B;
struct S_AB;
struct S_ABC;

impl A for S_A {}
impl B for S_B {}
impl A for S_AB {}
impl B for S_AB {}
impl A for S_ABC {}
impl B for S_ABC {}
impl C for S_ABC {}

fn main() {
    assert!(S_A.foo() == "implements_A_");
    assert!(S_B.foo() == "implements_B");
    assert!(S_AB.foo() == "implements_A+B");
    assert!(S_ABC.foo() == "implements_A+B+C");
}
