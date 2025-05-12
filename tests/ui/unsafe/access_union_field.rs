#![allow(unused_variables)]

union Foo {
    bar: i8,
    baz: u8,
}

fn main() {
    let foo = Foo { bar: 5 };
    let a = foo.bar; //~ ERROR access to union field is unsafe and requires unsafe function or block
    let b = foo.baz; //~ ERROR access to union field is unsafe and requires unsafe function or block
}
