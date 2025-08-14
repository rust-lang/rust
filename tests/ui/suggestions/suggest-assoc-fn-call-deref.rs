//@ run-rustfix

#![allow(unused)]

struct Foo<T>(T);

impl<T> Foo<T> {
    fn test() -> i32 { 1 }
}

fn main() {
    let x = Box::new(Foo(1i32));
    x.test();
    //~^ ERROR no method named `test` found for struct `Box<Foo<i32>>` in the current scope
}
