#![feature(type_alias_impl_trait)]

//@ check-pass

struct Foo<T>(T);

impl Foo<u32> {
    fn method() {}
    fn method2(self) {}
}

type Bar = impl Sized;

#[define_opaque(Bar)]
fn bar() -> Bar {
    42_u32
}

impl Foo<Bar> {
    #[define_opaque(Bar)]
    fn foo() -> Bar {
        Self::method();
        Foo::<Bar>::method();
        let x = Foo(bar());
        Foo::method2(x);
        let x = Self(bar());
        Self::method2(x);
        todo!()
    }
}

fn main() {}
