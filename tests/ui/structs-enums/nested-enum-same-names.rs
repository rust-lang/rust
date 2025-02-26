//@ run-pass
#![allow(dead_code)]

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
        enum Panic { Common }
    }
    pub fn bar() {
        enum Panic { Common }
    }
}

pub fn main() {}
