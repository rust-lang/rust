//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

trait Super<U> {
    fn a(&self) {
        let a: &dyn Sub = &();
        let b: &dyn Super<for<'a> fn(&'a ())> = a;
        //~^ ERROR the trait bound `&dyn Sub: CoerceUnsized<&dyn Super<for<'a> fn(&'a ())>>` is not satisfied
    }
}

impl<T> Super<T> for () {}

trait Sub: Super<fn(&'static ())> {}

impl Sub for () {}

fn main() {
    let a: &dyn Sub = &();
}
