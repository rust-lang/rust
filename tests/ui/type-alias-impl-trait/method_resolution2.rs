//! Check that we do unify `Bar<Foo>` with `Bar<u32>`, as the
//! `foo` method call can be resolved unambiguously by doing so.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(type_alias_impl_trait)]

type Foo = impl Sized;

struct Bar<T>(T);

impl Bar<Foo> {
    fn bar(self) {
        self.foo()
    }
}

impl Bar<u32> {
    fn foo(self) {}
}

fn foo() -> Foo {
    42_u32
}

fn main() {}
