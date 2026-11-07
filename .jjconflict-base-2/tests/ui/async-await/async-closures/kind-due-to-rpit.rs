//@ edition: 2024

// Make sure the error message is understandable when an `AsyncFn` goal is not satisfied
// (due to closure kind), and that goal originates from an RPIT.

fn repro(foo: impl Into<bool>) -> impl AsyncFn() {
    let inner_fn = async move || {
        //~^ ERROR expected a closure that implements the `AsyncFn` trait
        let _ = foo.into();
    };
    inner_fn
}

fn main() {}
