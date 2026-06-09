//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ edition: 2024

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/185>.
// Avoid unnecessarily computing the RPITIT type of the first impl when checking the WF of the
// second impl, since the first impl relies on the hidden type of the second impl.

trait Foo {
    fn call(self) -> impl Send;
}

trait Nested {}
impl<T> Foo for T
where
    T: Nested,
{
    fn call(self) -> impl Sized {
        NotSatisfied.call()
    }
}

struct NotSatisfied;
impl Foo for NotSatisfied {
    fn call(self) -> impl Sized {
        todo!()
    }
}

fn main() {}
