// build-pass (FIXME(62277): could be check-pass?)

#![deny(unused_variables)]

mod foo {
    enum Bar {}

    #[allow(dead_code)]
    pub struct Foo {
        value: Bar, // "privately" uninhabited
    }

    pub fn give_foo() -> Foo { panic!() }
}

fn main() {
    let a = 42;
    foo::give_foo();
    println!("Hello, {}", a); // ok: we can't tell that this code is dead
}
