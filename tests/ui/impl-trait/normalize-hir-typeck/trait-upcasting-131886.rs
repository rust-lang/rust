//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #131886
//@ compile-flags: -Zvalidate-mir
#![feature(type_alias_impl_trait)]

// This ICEd in MIR validation with the old solver. It's fixed by normalizing
// opaque types with the new solver. This should compile without a query cycle.

type Tait = impl Sized;

trait Foo: Bar<Tait> {}
trait Bar<T> {}

#[define_opaque(Tait)]
fn test_correct3(x: &dyn Foo) {
    _ = x as &dyn Bar<()>;
}

fn main() {}
