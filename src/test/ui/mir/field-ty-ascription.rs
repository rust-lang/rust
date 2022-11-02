// build-pass

struct Foo<T>(T); // `T` is covariant.

fn foo<'b>(x: Foo<for<'a> fn(&'a ())>) {
    let Foo(y): Foo<fn(&'b ())> = x;
}

fn foo_nested<'b>(x: Foo<Foo<for<'a> fn(&'a ())>>) {
    let Foo(Foo(y)): Foo<Foo<fn(&'b ())>> = x;
}

fn tuple<'b>(x: (u32, for<'a> fn(&'a ()))) {
    let (_, y): (u32, fn(&'b ())) = x;
}

fn tuple_nested<'b>(x: (u32, (u32, for<'a> fn(&'a ())))) {
    let (_, (_, y)): (u32, (u32, fn(&'b ()))) = x;
}

fn main() {}
