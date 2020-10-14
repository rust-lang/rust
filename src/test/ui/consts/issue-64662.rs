enum Foo {
    A = foo(), //~ ERROR: type annotations needed
    B = foo(), // We don't emit an error here, but if the one above is fixed, we will.
}

const fn foo<T>() -> isize {
    0
}

fn main() {}
