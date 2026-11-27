//@ needs-unwind
//@ compile-flags: -Zmir-opt-level=0 -Zverify-llvm-ir=yes
//@ check-pass

use std::ops::Deref;

trait Foo {
    fn method(&self) {}
}

impl Foo for u32 {}

fn via_deref_nested() -> Box<impl Deref<Target = impl Foo>> {
    if false {
        via_deref_nested().method();
    }

    Box::new(Box::new(1u32))
}

fn main() {
    via_deref_nested();
}
