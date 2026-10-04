// revisions: next old
//[next] compile-flags: -Znext-solver
//@ check-pass
// Regression test for https://github.com/rust-lang/rust/issues/154173.
// The ICE there was caused by a (flawed) attempt to eagerly normalize during generalization.
// The normalize would constrain other inference variables, which we couldn't deal with.

trait Trait<T> {
    type Assoc;
}

impl Trait<u32> for () {
    type Assoc = u32;
}

trait Eq {}
impl<C: Trait<T>, T> Eq for (C, T, <C as Trait<T>>::Assoc) {}
fn foo<A>()
where
    ((), A, A): Eq
{}

fn main() {
    foo::<_>();
}
