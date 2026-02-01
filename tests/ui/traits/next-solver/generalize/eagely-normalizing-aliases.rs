//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass
// Regression test for trait-system-refactor-initiative/262

trait View {}
trait HasAssoc {
    type Assoc;
}

struct StableVec<T>(T);
impl<T> View for StableVec<T> {}

fn assert_view<F: View>(f: F) -> F { f }


fn store<T>(x: StableVec<T::Assoc>)
where
    T: HasAssoc,
    StableVec<T>: View,
{
    let _: StableVec<T::Assoc> = assert_view(x);
}

fn main() {}
