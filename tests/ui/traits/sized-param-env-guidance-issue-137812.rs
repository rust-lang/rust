//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/137812>.
// Used to say the trait bound `T: Foo<(U,)>` is not satisfied
trait Foo<T> {
    fn method();
}

fn test<T, U>()
where
    T: Foo<(i32,)>,
    (U,): Sized,
{
    <T as Foo<(_,)>>::method();
}

fn main() {}
