//@ check-pass
//@ revisions: next assumptions
//@[next] compile-flags: -Znext-solver=globally
//@[assumptions] compile-flags: -Znext-solver=globally -Zassumptions-on-binders

// Reduced from the ecdsa build: matching an unnormalized item bound against
// a normalized environment must not reach the alias rigidness assertion.
trait Project {
    type Output;
}

trait Family {
    type A: Project;
    type B: Project<Output = Self::B>;
}

struct Holder<'a, T: Family>(&'a T::B);

fn check<'a, T: Family>(_: &'a T::B)
where
    <T::A as Project>::Output: 'a,
{}

trait Storage {
    type Repr: 'static;
}

struct Container<'a, T: Storage>(std::marker::PhantomData<&'a T::Repr>);

trait View<T> {
    type Item;
}

impl<'a, T: Storage + 'a> View<T> for Container<'a, T> {
    type Item = T;
}

trait Key<'a>: Storage + Sized {
    type Container: View<Self, Item = Self>;
}

// An unused item-bound equality must not introduce the impl's `T: 'a` bound.
fn unrelated<'a, T: Key<'a, Container = Container<'a, T>>>(
    value: Container<'a, T>,
) -> Container<'a, T> {
    value
}

fn main() {}
