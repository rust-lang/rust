trait Bound {}
struct NeedsBound<T: Bound>(T);

// Checks that we enforce that closure args are WF.

fn constrain_inner<T, F: for<'a> FnOnce(&'a (), NeedsBound<T>)>(_: T, _: F) {}

fn main() {
    constrain_inner(1u32, |&(), _| ());
    //~^ ERROR the trait bound `u32: Bound` is not satisfied
}
