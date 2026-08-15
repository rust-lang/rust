//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

// #136824 changed cycles to be coinductive if they have at least
// one productive step, causing this test to pass with the new solver.
//
// The cycle in the test is the following:
// - `Foo<T>: Send`, builtin auto-trait impl requires
// - `<Foo<T> as Trait>::Assoc: Send`, requires normalizing self type via impl, requires
// - `Foo<T>: SendIndir`, via impl requires
// - `Foo<T>: Send` cycle
//
// The old solver treats this cycle as inductive due to the `Foo<T>: SendIndir` step.

struct Foo<T>(<Foo<T> as Trait>::Assoc);
//[current]~^ ERROR overflow evaluating the requirement `Foo<T>: SendIndir`

trait SendIndir {}
impl<T: Send> SendIndir for T {}

trait Trait {
    type Assoc;
}
impl<T: SendIndir> Trait for T {
    type Assoc = ();
}

fn is_send<T: Send>() {}

fn main() {
    is_send::<Foo<u32>>();
}
