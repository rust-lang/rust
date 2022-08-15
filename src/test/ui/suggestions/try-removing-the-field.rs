// run-pass

#![allow(dead_code)]

struct Foo {
    foo: i32,
    bar: (),
    baz: (),
}

fn use_foo(x: Foo) -> i32 {
    let Foo { foo, bar, .. } = x; //~ WARNING unused variable: `bar`
                                  //~| help: try removing the field
    return foo;
}

fn main() {}
