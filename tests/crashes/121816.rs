//@ known-bug: #121816
fn f<'a, T>(_: &'static &'a (), x: &'a T) -> &'static T {
    x
}
trait W<'a> {
    fn g<T>(self, x: &'a T) -> &'static T;
}
impl<'a> W<'a> for &'static () {
    fn g<T>(self, x: &'a T) -> &'static T {
        f(&self, x)
    }
}
