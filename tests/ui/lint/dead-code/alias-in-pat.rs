//@ run-pass

#![deny(dead_code)]

fn main() {
    struct Foo<T> { x: T }
    type Bar = Foo<u32>;
    let spam = |Bar { x }| x != 0;
    println!("{}", spam(Foo { x: 10 }));
}
