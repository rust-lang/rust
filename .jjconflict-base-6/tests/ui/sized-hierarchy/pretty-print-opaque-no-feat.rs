//@ compile-flags: --crate-type=lib

pub trait Tr {}
impl Tr for u32 {}

pub fn foo() -> Box<impl Tr + ?Sized> {
    if true {
        let x = foo();
        let y: Box<dyn Tr> = x;
//~^ ERROR: the size for values of type `impl Tr + ?Sized` cannot be known
    }
    Box::new(1u32)
}
