// This example incorrectly compiled while refactoring the way
// closure requirements are handled.

struct Outlives<'hr: 'scope, 'scope>(*mut (&'scope (), &'hr ()));
impl<'hr, 'scope> Outlives<'hr, 'scope> {
    fn outlives_hr<T: 'hr>(self) {}
}

fn takes_closure_implied_bound<'scope>(f: impl for<'hr> FnOnce(Outlives<'hr, 'scope>)) {}

fn requires_external_outlives_hr<T>() {
    // implied bounds:
    // - `T: 'scope` as `'scope` is local to this function
    // - `'hr: 'scope` as it's an implied bound of `Outlives`
    //
    // need to prove `T: 'hr` :<
    takes_closure_implied_bound(|proof| proof.outlives_hr::<T>());
    //~^ ERROR the parameter type `T` may not live long enough
}

fn main() {}
