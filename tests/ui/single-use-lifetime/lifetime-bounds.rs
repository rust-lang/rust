// Regression test for https://github.com/rust-lang/rust/issues/153836.
#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

struct Foo<'a>(&'a i32);
struct Bar<'b>(&'b i32);

fn function<'a, 'b: 'a>(_: Foo<'a>, _: Bar<'b>) {}

type FnPtr = for<'a, 'b: 'a> fn(Foo<'a>, Bar<'b>);
//~^ ERROR bounds cannot be used in this context

trait WhereBound<'a, 'b> {}

fn where_bound<T>()
where
    T: for<'a, 'b: 'a> WhereBound<'a, 'b>,
    //~^ ERROR bounds cannot be used in this context
{
}

trait ImplTrait<'a> {
    fn foo(self, foo: Foo<'a>);
}

impl<'a, 'b: 'a> ImplTrait<'a> for Bar<'b> {
    fn foo(self, foo: Foo<'a>) {
        let _: &'a i32 = self.0;
        let _: Foo<'a> = foo;
    }
}

fn main() {}
