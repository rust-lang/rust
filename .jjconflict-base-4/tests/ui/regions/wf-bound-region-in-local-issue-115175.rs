// A regression test for https://github.com/rust-lang/rust/issues/115175.
// This used to compile without error despite of unsatisfied outlives bound `T: 'static` on local.

struct Static<T: 'static>(T);

fn test<T>() {
    let _ = None::<Static<T>>;
    //~^ ERROR the parameter type `T` may not live long enough
}

fn main() {}
