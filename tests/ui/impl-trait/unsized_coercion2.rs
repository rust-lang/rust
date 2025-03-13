//! This test checks that opaque types get unsized instead of
//! constraining their hidden type to a trait object.

//@ revisions: next old
//@[next] compile-flags: -Znext-solver
//@ check-pass

trait Trait {}

impl Trait for u32 {}

fn hello() -> Box<impl Trait + ?Sized> {
    if true {
        let x = hello();
        let y: Box<dyn Trait> = x;
    }
    Box::new(1u32)
}

fn main() {}
