trait foo<T> { }

fn bar(x: foo<uint>) -> foo<int> {
    return (x as foo::<int>);
    //~^ ERROR mismatched types: expected `@foo<int>` but found `@foo<uint>`
    //~^^ ERROR mismatched types: expected `@foo<int>` but found `@foo<uint>`
    // This is unfortunate -- new handling of parens means the error message
    // gets printed twice
}

fn main() {}
