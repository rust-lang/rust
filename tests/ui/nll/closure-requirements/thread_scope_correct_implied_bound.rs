// This example broke while refactoring the way closure
// requirements are handled. The setup here matches
// `thread::scope`.

//@ check-pass

struct Outlives<'hr, 'scope: 'hr>(*mut (&'scope (), &'hr ()));
impl<'hr, 'scope> Outlives<'hr, 'scope> {
    fn outlives_hr<T: 'hr>(self) {}
}

fn takes_closure_implied_bound<'scope>(f: impl for<'hr> FnOnce(Outlives<'hr, 'scope>)) {}

fn requires_external_outlives_hr<T>() {
    // implied bounds:
    // - `T: 'scope` as `'scope` is local to this function
    // - `'scope: 'hr` as it's an implied bound of `Outlives`
    //
    // need to prove `T: 'hr` :>
    takes_closure_implied_bound(|proof| proof.outlives_hr::<T>());
}

fn main() {}
