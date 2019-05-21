// run-pass

#![deny(unused_mut)]

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
