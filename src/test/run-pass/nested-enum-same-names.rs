// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*

#7770 ICE with sibling methods containing same-name-enum containing
 same-name-member

If you have two methods in an impl block, each containing an enum
(with the same name), each containing at least one value with the same
name, rustc gives the same LLVM symbol for the two of them and fails,
as it does not include the method name in the symbol name.

*/

pub struct Foo;
impl Foo {
    pub fn foo() {
        enum Panic { Common };
    }
    pub fn bar() {
        enum Panic { Common };
    }
}

pub fn main() {}
