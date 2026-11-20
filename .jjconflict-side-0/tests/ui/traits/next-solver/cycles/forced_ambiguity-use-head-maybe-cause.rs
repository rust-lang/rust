//@ compile-flags: -Znext-solver
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

// A regression test making sure that when forcing dependent
// provisional cache entries to ambiguous, we use the `MaybeCause`
// of the cycle head. We ended up trying to use the current result
// of the provisional cache entry, which is incorrect and caused an
// ICE when trying to unwrap it.

struct Root<T>(T);
struct Head<T>(T);
struct Error<T>(T);
struct NotImplemented<T>(T);

#[rustc_coinductive]
trait Trait {}
impl<T> Trait for Root<T>
where
    Head<T>: Trait,
{}

impl<T> Trait for Head<T>
where
    Root<T>: Trait,
    T: Trait, // ambiguous
{}

impl<T> Trait for Head<T>
where
    Error<T>: Trait,
    NotImplemented<T>: Trait,
{}

impl<T> Trait for Error<T>
where
    Head<T>: Trait,
    NotImplemented<T>: Trait,
{}

fn impls_trait<T: Trait>() {}
fn main() {
    impls_trait::<Root<_>>() //~ ERROR type annotations needed
}
