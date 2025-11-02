//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#191.

trait Indir<T>: FnOnce(T) -> Self::Ret {
    type Ret;
}
impl<F, T, R> Indir<T> for F where F: FnOnce(T) -> R {
    type Ret = R;
}

trait Mirror {
    type Assoc<'a>;
}

fn needs<T: Mirror>(_: impl for<'a> Indir<T::Assoc<'a>>) {}

fn test<T>() where for<'a> T: Mirror<Assoc<'a> = i32> {
    needs::<T>(|x| { x.to_string(); });
}
fn main() {}
