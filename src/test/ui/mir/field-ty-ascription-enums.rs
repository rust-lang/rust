// build-pass

enum Foo<T> {
    Var(T),
} // `T` is covariant.

fn foo<'b>(x: Foo<for<'a> fn(&'a ())>) {
    let Foo::Var(x): Foo<fn(&'b ())> = x;
}

fn foo_nested<'b>(x: Foo<Foo<for<'a> fn(&'a ())>>) {
    let Foo::Var(Foo::Var(x)): Foo<Foo<fn(&'b ())>> = x;
}

fn main() {}
