// compile-pass

#![deny(unreachable_code)]

mod foo {
    enum Bar {}

    #[allow(dead_code)]
    pub struct Foo {
        value: Bar, // "privately" uninhabited
    }

    pub fn give_foo() -> Foo { panic!() }
}

fn main() {
    foo::give_foo();
    println!("Hello, world!"); // ok: we can't tell that this code is dead
}
