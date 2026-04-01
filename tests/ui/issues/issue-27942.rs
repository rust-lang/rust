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
