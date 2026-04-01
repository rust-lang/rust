//@ check-pass
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// Make sure that refinement checking doesn't cause a cycle in `Instance::resolve`
// which calls `compare_impl_item`.

trait Foo {
    fn test() -> impl IntoIterator<Item = ()> + Send;
}

struct A;
impl Foo for A {
    fn test() -> impl IntoIterator<Item = ()> + Send {
        B::test()
    }
}

struct B;
impl Foo for B {
    fn test() -> impl IntoIterator<Item = ()> + Send {
        for () in A::test() {}

        []
    }
}

fn main() {}
