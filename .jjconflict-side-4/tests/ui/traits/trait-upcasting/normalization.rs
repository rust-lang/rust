//@ check-pass
// issue: 114113
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

trait Bar<T> {}
trait Foo<T>: Bar<<T as Mirror>::Assoc> {}

fn upcast<T>(x: &dyn Foo<T>) -> &dyn Bar<T> { x }

fn main() {}
