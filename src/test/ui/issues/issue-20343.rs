// run-pass
#![allow(unused_variables)]
// Regression test for Issue #20343.

// pretty-expanded FIXME #23616

#![deny(dead_code)]

struct B { b: u32 }
struct C;
struct D;

trait T<A> { fn dummy(&self, a: A) { } }
impl<A> T<A> for () {}

impl B {
    // test for unused code in arguments
    fn foo(B { b }: B) -> u32 { b }

    // test for unused code in return type
    fn bar() -> C { unsafe { ::std::mem::transmute(()) } }

    // test for unused code in generics
    fn baz<A: T<D>>() {}
}

pub fn main() {
    let b = B { b: 3 };
    B::foo(b);
    B::bar();
    B::baz::<()>();
}
