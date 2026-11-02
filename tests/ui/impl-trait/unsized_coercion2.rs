//! This test checks that opaque types get unsized instead of
//! constraining their hidden type to a trait object.

//@ revisions: next old
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

trait Trait {}

impl Trait for u32 {}

fn hello() -> Box<impl Trait + ?Sized> {
    if true {
        let x = hello();
        let y: Box<dyn Trait> = x;
        //[old]~^ ERROR: the size for values of type `impl Trait + ?Sized` cannot be know
    }
    Box::new(1u32)
}

fn main() {}
