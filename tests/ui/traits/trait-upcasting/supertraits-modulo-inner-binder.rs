//@ run-pass

trait Super<U> {
    fn call(&self)
    where
        U: HigherRanked,
    {
    }
}

impl<T> Super<T> for () {}

trait HigherRanked {}
impl HigherRanked for for<'a> fn(&'a ()) {}

trait Unimplemented {}
impl<T: Unimplemented> HigherRanked for T {}

trait Sub: Super<fn(&'static ())> + Super<for<'a> fn(&'a ())> {}
impl Sub for () {}

fn main() {
    let a: &dyn Sub = &();
    // `Super<fn(&'static ())>` and `Super<for<'a> fn(&'a ())>` have different
    // vtables and we need to upcast to the latter!
    let b: &dyn Super<for<'a> fn(&'a ())> = a;
    b.call();
}
