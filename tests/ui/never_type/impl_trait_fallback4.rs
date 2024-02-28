#![feature(type_alias_impl_trait)]

trait T {
    type Assoc: Cake;
}

trait Cake: std::fmt::Display {
    fn cake() -> Self;
}

type Foo = impl T;

fn foo() -> impl T {
    //~^ ERROR trait `T` is not implemented for `()`
    panic!()
}

fn a() -> Foo {
    foo()
}

fn main() {
    println!("{}", <Foo as T>::Assoc::cake());
}
