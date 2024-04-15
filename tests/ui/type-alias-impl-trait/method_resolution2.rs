//! Check that we do not unify `Bar<Foo>` with `Bar<u32>`, even though the
//! `foo` method call can be resolved unambiguously by doing so.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

#![feature(type_alias_impl_trait)]

type Foo = impl Sized;
//[current]~^ ERROR: cycle

struct Bar<T>(T);

impl Bar<Foo> {
    fn bar(self) {
        self.foo()
        //[current]~^ ERROR: no method named `foo`
    }
}

impl Bar<u32> {
    fn foo(self) {}
}

fn foo() -> Foo {
    42_u32
}

fn main() {}
