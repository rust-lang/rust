trait foo { fn foo(); }

fn to_foo<T: Copy foo>(t: T) -> foo {
    t as foo //~ ERROR value may contain borrowed pointers; use `owned` bound
}

fn to_foo2<T: Copy foo Owned>(t: T) -> foo {
    t as foo
}

fn main() {}
