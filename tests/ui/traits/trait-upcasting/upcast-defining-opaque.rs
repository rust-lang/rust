//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@check-pass

#![feature(trait_upcasting, type_alias_impl_trait)]

trait Super {
    type Assoc;
}

trait Sub: Super {}

impl<T: ?Sized> Super for T {
    type Assoc = i32;
}

type Foo = impl Sized;

fn upcast(x: &dyn Sub<Assoc = Foo>) -> &dyn Super<Assoc = i32> {
    x
}

fn main() {}
