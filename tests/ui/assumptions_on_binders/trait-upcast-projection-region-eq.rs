//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

trait Super {
    type Assoc;

    fn a(&self) {
        let a: &dyn Sub = &();
        let b: &dyn Super<Assoc = for<'a> fn(&'a ())> = a;
        //~^ ERROR the trait bound `&dyn Sub: CoerceUnsized<&dyn Super<Assoc = for<'a> fn(&'a ())>>` is not satisfied
    }
}

impl Super for () {
    type Assoc = fn(&'static ());
}

trait Sub: Super<Assoc = fn(&'static ())> {}

impl Sub for () {}

fn main() {
    let a: &dyn Sub = &();
}
