//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

// #136824 changed cycles to be coinductive if they have at least
// one productive step, causing this test to pass with the new solver.
//
// The cycle in the test is the following:
// - `Foo<T>: Send`, impl requires
// - `T: SendIndir<Foo<T>>`, impl requires
// - `Foo<T>: Send` cycle
//
// The old solver treats this cycle as inductive due to the `T: SendIndir` step.

struct Foo<T>(T);
unsafe impl<T: SendIndir<Foo<T>>> Send for Foo<T> {}

trait SendIndir<T> {}
impl<T, U: Send> SendIndir<U> for T {}

fn is_send<T: Send>() {}
fn main() {
    is_send::<Foo<u32>>();
    //[current]~^ ERROR overflow evaluating the requirement `u32: SendIndir<Foo<u32>>`
}
