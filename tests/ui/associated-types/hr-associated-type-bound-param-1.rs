trait Y<'a, T: ?Sized>
where
    T: Y<'a, Self>,
    for<'b> <Self as Y<'b, T>>::V: Clone,
    for<'b> <T as Y<'b, Self>>::V: Clone,
{
    type V: ?Sized;
    fn g(&self, x: &Self::V) {
        <Self::V>::clone(x);
    }
}

impl<'a> Y<'a, u8> for u8 {
    type V = str;
    //~^ ERROR trait `Clone` is not implemented for `str`
}

fn main() {
    1u8.g("abc");
    //~^ ERROR trait `Clone` is not implemented for `str`
}
