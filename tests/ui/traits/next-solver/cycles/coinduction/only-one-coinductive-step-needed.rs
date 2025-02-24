//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

// #136824 changed cycles to be coinductive if they have at least
// one productive step, causing this test to pass with the new solver.

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
