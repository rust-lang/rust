//! Check that one cannot use arbitrary self types where a generic parameter
//! mismatches with an opaque type. In theory this could unify with the opaque
//! type, registering the generic parameter as the hidden type of the opaque type.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver

#![feature(type_alias_impl_trait, arbitrary_self_types)]

type Foo = impl Copy;

#[derive(Copy, Clone)]
struct Bar<T>(T);

impl Bar<Foo> {
    fn bar(self: Bar<u32>) {
        //~^ ERROR: invalid `self` parameter
        self.foo()
    }
    fn baz(self: &Bar<u32>) {
        //~^ ERROR: invalid `self` parameter
        self.foo()
    }
}

impl Bar<u32> {
    fn foo(self) {}
}

#[define_opaque(Foo)]
fn foo() -> Foo {
    42_u32
}

fn main() {}
