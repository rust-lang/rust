// Don't ICE when trying to annotate signature and we see `&()`

fn f<'a, T>(_: &'static &'a (), x: &'a T) -> &'static T {
    x
}
trait W<'a> {
    fn g<T>(self, x: &'a T) -> &'static T;
}

// Frankly this error message is impossible to parse, but :shrug:.
impl<'a> W<'a> for &'static () {
    fn g<T>(self, x: &'a T) -> &'static T {
        f(&self, x)
        //~^ ERROR borrowed data escapes outside of method
        //~| ERROR `self` does not live long enough
    }
}

fn main() {}
