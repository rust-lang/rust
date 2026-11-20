trait ATC<'a> {
    type Type: Sized;
}

trait WithDefault: for<'a> ATC<'a> {
    fn with_default<F: for<'a> Fn(<Self as ATC<'a>>::Type)>(f: F);
}

fn call<'b, T: for<'a> ATC<'a>, F: for<'a> Fn(<T as ATC<'a>>::Type)>(
    f: F,
    x: <T as ATC<'b>>::Type,
) {
    f(x);
}

impl<'a> ATC<'a> for () {
    type Type = Self;
}

impl WithDefault for () {
    fn with_default<F: for<'a> Fn(<Self as ATC<'a>>::Type)>(f: F) {
        // Errors with a bogus type mismatch.
        //f(());
        // Going through another generic function works fine.
        call(f, ());
        //~^ ERROR expected a
    }
}

fn main() {
    // <()>::with_default(|_| {});
}
