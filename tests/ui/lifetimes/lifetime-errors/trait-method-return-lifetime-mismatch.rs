//! Regression test for <https://github.com/rust-lang/rust/issues/27942>.
//! Test internal compiler structs do not leak into error message.
//@ dont-require-annotations: NOTE

pub trait Resources<'a> {}

pub trait Buffer<'a, R: Resources<'a>> {

    fn select(&self) -> BufferViewHandle<R>;
    //~^ ERROR mismatched types
    //~| NOTE lifetime mismatch
    //~| ERROR mismatched types
    //~| NOTE lifetime mismatch
}

pub struct BufferViewHandle<'a, R: 'a+Resources<'a>>(&'a R);

fn main() {}
