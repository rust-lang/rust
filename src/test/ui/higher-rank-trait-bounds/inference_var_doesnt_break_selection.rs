// check-pass
trait Foo {}
impl Foo for bool {}
impl Foo for (bool, bool) {}

trait Bar<'a> {
    type Assoc;
}
impl<'a> Bar<'a> for u8 {
    type Assoc = bool;
}
impl<'a, T: Bar<'a>, U: Bar<'a>> Bar<'a> for (T, U) {
    type Assoc = (T::Assoc, U::Assoc);
}

fn calls() {
    has_bound((1_u8, 1_u8));
}

fn has_bound<T>(_: T)
where
    for<'a> T: Bar<'a>,
    for<'a> <T as Bar<'a>>::Assoc: Foo,
{
}

fn main() {
    calls()
}
