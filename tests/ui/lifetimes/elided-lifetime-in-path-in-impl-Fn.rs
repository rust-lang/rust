//@ check-pass

struct Foo<'a>(&'a ());

fn with_fn() -> fn(Foo) {
    |_| ()
}

fn with_impl_fn() -> impl Fn(Foo) {
    |_| ()
}

fn with_where_fn<T>()
where
    T: Fn(Foo),
{
}

fn main() {}
