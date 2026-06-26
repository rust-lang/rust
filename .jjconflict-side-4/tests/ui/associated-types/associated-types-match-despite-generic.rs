//! Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/216>.
//@compile-flags: -Znext-solver=globally
//@ check-pass

struct Outer;
struct Inner;
trait Id<T> {
    type This;
}
impl<T, U> Id<U> for T {
    type This = T;
}

fn free<T>(x: T) -> <T as Id<Inner>>::This
where
    <T as Id<Outer>>::This: Id<Inner, This = <T as Id<Inner>>::This>,
{
    x
}

fn main() {}
