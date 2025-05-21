//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver


// Make sure that having an applicable user-written
// and builtin impl is ambiguous.

trait Equals<T: ?Sized> {}

impl<T: ?Sized> Equals<T> for T {}

fn impls_equals<T: Equals<U> + ?Sized, U: ?Sized>() {}

fn main() {
    impls_equals::<dyn Equals<u32>, _>();
    //~^ ERROR type annotations needed
}
