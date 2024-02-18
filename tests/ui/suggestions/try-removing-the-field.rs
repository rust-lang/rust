//@ run-pass

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

// issue #105028, suggest removing the field only for shorthand
fn use_match(x: Foo) {
    match x {
        Foo { foo: unused, .. } => { //~ WARNING unused variable
                                     //~| help: if this is intentional, prefix it with an underscore
        }
    }

    match x {
        Foo { foo, .. } => { //~ WARNING unused variable
                             //~| help: try removing the field
        }
    }
}

fn main() {}
