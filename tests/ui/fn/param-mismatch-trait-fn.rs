trait Foo {
    fn same_type<T>(_: T, _: T);
}

fn f<T: Foo, X, Y>(x: X, y: Y) {
    T::same_type([x], Some(y));
    //~^ ERROR mismatched types
}

fn main() {}
