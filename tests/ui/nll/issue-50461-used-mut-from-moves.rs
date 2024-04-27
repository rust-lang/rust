//@ run-pass

#![deny(unused_mut)]
#![allow(dead_code)]

struct Foo {
    pub value: i32
}

fn use_foo_mut(mut foo: Foo) {
    foo = foo;
    println!("{}", foo.value);
}

fn main() {
    use_foo_mut(Foo { value: 413 });
}
