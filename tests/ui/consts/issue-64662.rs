enum Foo {
    A = foo(), //~ ERROR: type annotations needed
    B = foo(), //~ ERROR: type annotations needed
}

const fn foo<T>() -> isize {
    0
}

fn main() {}
