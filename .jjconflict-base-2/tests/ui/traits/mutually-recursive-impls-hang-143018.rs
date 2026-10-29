//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/143018>.

trait Trait<V> {}
struct A;
struct B;

impl<V> Trait<V> for A
where
    A: Trait<V>,
    B: Trait<V>,
{
}

impl<V> Trait<V> for B
where
    A: Trait<V>,
{
}

fn impls_trait<T: Trait<V>, V>() {}
fn main() {
    impls_trait::<A, _>();
    //[current]~^ ERROR type annotations needed
    //[next]~^^ ERROR overflow evaluating the requirement
}
