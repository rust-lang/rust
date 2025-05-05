//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/194>.
// Ensure that we check the well-formedness of `<Check as Mode>::Output<T>` after normalizing
// the type to `()`, since we only imply outlives bounds from the normalized signature, so we
// don't know (e.g.) that `&mut T` is WF.


trait Mode {
    type Output<T>;
    fn from_mut<T>(_r: &mut Self::Output<T>) -> Self::Output<&mut T>;
}

struct Check;

impl Mode for Check {
    type Output<T> = ();
    fn from_mut<T>(_r: &mut Self::Output<T>) -> Self::Output<&mut T> {}
}

fn main() {}
