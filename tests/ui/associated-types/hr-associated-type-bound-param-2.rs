trait Z<'a, T: ?Sized>
where
    T: Z<'a, u16>,
    //~^ ERROR trait `Clone` is not implemented for `str`
    //~| ERROR trait `Clone` is not implemented for `str`
    for<'b> <T as Z<'b, u16>>::W: Clone,
{
    type W: ?Sized;
    fn h(&self, x: &T::W) {
        <T::W>::clone(x);
        //~^ ERROR trait `Clone` is not implemented for `str`
        //~| ERROR trait `Clone` is not implemented for `str`
    }
}

impl<'a> Z<'a, u16> for u16 {
    type W = str;
    //~^ ERROR the trait `Clone` is not implemented for `str`
}

fn main() {
    1u16.h("abc");
    //~^ ERROR the trait `Clone` is not implemented for `str`
}
