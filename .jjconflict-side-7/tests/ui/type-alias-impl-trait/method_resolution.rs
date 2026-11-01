//! `Bar<u32>::foo` is not defining `Foo`, so it cannot rely on the fact that
//! `u32` is the hidden type of `Foo` to call `bar`

//@ revisions: current next
//@[next] compile-flags: -Znext-solver

#![feature(type_alias_impl_trait)]

type Foo = impl Sized;

struct Bar<T>(T);

impl Bar<Foo> {
    #[define_opaque(Foo)]
    fn bar(mut self) {
        self.0 = 42_u32;
    }
}

impl Bar<u32> {
    fn foo(self) {
        self.bar()
        //~^ ERROR: no method named `bar`
    }
}

#[define_opaque(Foo)]
fn foo() -> Foo {
    42_u32
}

fn main() {}
