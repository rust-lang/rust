//! Even though `Bar<u32>::foo` is defining `Foo`, the old solver does
//! not figure out that `u32` is the hidden type of `Foo` to call `bar`.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(type_alias_impl_trait)]

type Foo = impl Sized;

struct Bar<T>(T);

impl Bar<Foo> {
    #[defines(Foo)]
    fn bar(mut self) {
        self.0 = 42_u32;
    }
}

impl Bar<u32> {
    #[defines(Foo)]
    fn foo(self) {
        self.bar()
    }
}

#[defines(Foo)]
fn foo() -> Foo {
    42_u32
}

fn main() {}
