enum Foo<T> {
    Var(T),
} // `T` is covariant.

fn foo<'b>(x: Foo<for<'a> fn(&'a ())>) {
    let Foo::Var(x): Foo<fn(&'b ())> = x;
    //~^ ERROR lifetime may not live long enough
    //~| ERROR mismatched types
    //~| ERROR mismatched types
}

fn foo_nested<'b>(x: Foo<Foo<for<'a> fn(&'a ())>>) {
    let Foo::Var(Foo::Var(x)): Foo<Foo<fn(&'b ())>> = x;
    //~^ ERROR lifetime may not live long enough
    //~| ERROR mismatched types
    //~| ERROR mismatched types
}

fn main() {}
