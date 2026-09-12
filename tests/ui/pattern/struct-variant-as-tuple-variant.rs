//! Regression test for <https://github.com/rust-lang/rust/issues/19086>.

use Foo::FooB;

enum Foo {
    FooB { x: i32, y: i32 }
}

fn main() {
    let f = FooB { x: 3, y: 4 };
    match f {
        FooB(a, b) => println!("{} {}", a, b),
        //~^ ERROR cannot find tuple struct or tuple variant `FooB` in this scope
    }
}
