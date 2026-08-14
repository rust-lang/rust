//@ known-bug: rust-lang/rust#142866
//@ compile-flags: -Znext-solver=globally
trait Trait<T> {}
struct A<T>(T);
struct B<T>(T);

trait IncompleteGuidance {}

impl<T> Trait<()> for A<T>
where
    T: IncompleteGuidance,
{
}

impl<T, U> Trait<()> for B<T>
where
    A<T>: Trait<U>,
{
}

fn impls_trait<T: Trait<()>>() {}

fn main() {
    impls_trait::<B<()>>();
}
