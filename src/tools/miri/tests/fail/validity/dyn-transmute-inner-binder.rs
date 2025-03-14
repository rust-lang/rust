// Test that transmuting from `&dyn Trait<fn(&'static ())>` to `&dyn Trait<for<'a> fn(&'a ())>` is UB.
//
// The vtable of `() as Trait<fn(&'static ())>` and `() as Trait<for<'a> fn(&'a ())>` can have
// different entries and, because in the former the entry for `foo` is vacant, this test will
// segfault at runtime.

trait Trait<U> {
    fn foo(&self)
    where
        U: HigherRanked,
    {
    }
}
impl<T, U> Trait<U> for T {}

trait HigherRanked {}
impl HigherRanked for for<'a> fn(&'a ()) {}

// 2nd candidate is required so that selecting `(): Trait<fn(&'static ())>` will
// evaluate the candidates and fail the leak check instead of returning the
// only applicable candidate.
trait Unsatisfied {}
impl<T: Unsatisfied> HigherRanked for T {}

fn main() {
    let x: &dyn Trait<fn(&'static ())> = &();
    let y: &dyn Trait<for<'a> fn(&'a ())> = unsafe { std::mem::transmute(x) };
    //~^ ERROR: wrong trait in wide pointer vtable
    y.foo();
}
