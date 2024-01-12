// We don't infer `T: 'static` outlives relationships.

// check-pass

struct Foo<U> {
    bar: Bar<U>,
}
struct Bar<T: 'static> {
    x: T,
}

fn main() {}
