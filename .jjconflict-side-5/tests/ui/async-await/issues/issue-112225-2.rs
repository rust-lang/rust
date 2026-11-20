//@ edition:2021

// With the current compiler logic, we cannot have both the `112225-1` case,
// and this `112225-2` case working, as the type inference depends on the evaluation
// order, and there is some explicit ordering going on.
// See the `check_closures` part in `FnCtxt::check_argument_types`.
// The `112225-1` case was a regression in real world code, whereas the `112225-2`
// case never used to work prior to 1.70.

use core::future::Future;

fn main() {
    let x = Default::default();
    //~^ ERROR: type annotations needed
    do_async(
        async { x.0; },
        { || { let _: &(i32,) = &x; } },
    );
}
fn do_async<Fut, T>(_fut: Fut, _val: T, ) {}
