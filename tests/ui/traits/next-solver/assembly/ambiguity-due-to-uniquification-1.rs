//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for #139409 and trait-system-refactor-initiative#27.

trait B<C> {}
impl<C> B<C> for () {}
trait D<C, E>: B<C> + B<E> {
    fn f(&self) {}
}
impl<C, E> D<C, E> for () {}
fn main() {
    (&() as &dyn D<&(), &()>).f()
}
