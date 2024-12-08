// The method `assert_static` should be callable only for static values,
// because the impl has an implied bound `where T: 'static`.

//@ check-fail

trait AnyStatic<Witness>: Sized {
    fn assert_static(self) {}
}

impl<T> AnyStatic<&'static T> for T {}

fn main() {
    (&String::new()).assert_static();
    //~^ ERROR temporary value dropped while borrowed
}
