//@ ignore-compare-mode-next-solver
//@ compile-flags: -Znext-solver
//@ check-pass
#![feature(type_alias_impl_trait)]

// Make sure that we support non-defining uses in HIR typeck.
// Regression test for trait-system-refactor-initiative#135.

fn non_defining_recurse<T>(b: bool) -> impl Sized {
    if b {
        // This results in an opaque type use `opaque<()> = ?unconstrained`
        // during HIR typeck.
        non_defining_recurse::<()>(false);
    }
}

trait Eq<T, U> {}
impl<T> Eq<T, T> for () {}
fn is_eq<T: Eq<U, V>, U, V>() {}
type Tait<T> = impl Sized;
#[define_opaque(Tait)]
fn non_defining_explicit<T>() {
    is_eq::<(), Tait<_>, u32>(); // constrains opaque type args via hidden type
    is_eq::<(), Tait<u64>, _>(); // constraints hidden type via args
    is_eq::<(), Tait<T>, T>(); // actually defines
}

fn main() {}
