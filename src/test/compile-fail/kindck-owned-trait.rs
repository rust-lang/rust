trait foo { fn foo(); }

fn to_foo<T: copy foo>(t: T) -> foo {
    t as foo //~ ERROR value may contain borrowed pointers; use `owned` bound
}

fn to_foo2<T: copy foo owned>(t: T) -> foo {
    t as foo
}

fn main() {}
