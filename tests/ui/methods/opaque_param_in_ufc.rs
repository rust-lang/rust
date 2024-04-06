#![feature(type_alias_impl_trait)]
struct Foo<T>(T);

impl Foo<u32> {
    fn method() {}
    fn method2(self) {}
}

type Bar = impl Sized;

fn bar() -> Bar {
    42_u32
}

impl Foo<Bar> {
    fn foo() -> Bar {
        Self::method();
        //~^ ERROR: no function or associated item named `method` found for struct `Foo<Bar>`
        Foo::<Bar>::method();
        //~^ ERROR: no function or associated item named `method` found for struct `Foo<Bar>`
        let x = Foo(bar());
        Foo::method2(x);
        let x = Self(bar());
        Self::method2(x);
        //~^ ERROR: no function or associated item named `method2` found for struct `Foo<Bar>`
        todo!()
    }
}

fn main() {}
