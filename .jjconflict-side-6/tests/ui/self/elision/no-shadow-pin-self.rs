use std::pin::Pin;
trait Trait {
    fn method<'a>(self: Pin<&Self>, f: &'a u32) -> &'a u32 {
        f
    }
}

impl<P> Trait for Pin<P> {
    // This should not hide `&Self`, which would cause this to compile.
    fn method(self: Pin<&Self>, f: &u32) -> &u32 {
        //~^ ERROR `impl` item signature doesn't match `trait`
        f
        //~^ ERROR lifetime may not live long enough
    }
}

fn main() {}
