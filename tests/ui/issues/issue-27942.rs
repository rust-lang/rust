pub trait Resources<'a> {}

pub trait Buffer<'a, R: Resources<'a>> {

    fn select(&self) -> BufferViewHandle<R>;
    //~^ ERROR mismatched types
    //~| lifetime mismatch
    //~| ERROR mismatched types
    //~| lifetime mismatch
}

pub struct BufferViewHandle<'a, R: 'a+Resources<'a>>(&'a R);

fn main() {}
