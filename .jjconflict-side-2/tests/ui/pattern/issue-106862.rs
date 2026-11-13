//@ run-rustfix

#![allow(unused)]

use Foo::{FooB, FooA};

enum Foo {
    FooA { opt_x: Option<i32>, y: i32 },
    FooB { x: i32, y: i32 }
}

fn main() {
    let f = FooB { x: 3, y: 4 };

    match f {
        FooB(a, b) => println!("{} {}", a, b),
        //~^ ERROR cannot find tuple struct or tuple variant `FooB` in this scope
        _ => (),
    }

    match f {
        FooB(x, y) => println!("{} {}", x, y),
        //~^ ERROR cannot find tuple struct or tuple variant `FooB` in this scope
        _ => (),
    }

    match f {
        FooA(Some(x), y) => println!("{} {}", x, y),
        //~^ ERROR cannot find tuple struct or tuple variant `FooA` in this scope
        _ => (),
    }

    match f {
        FooB(a, _, _) => println!("{}", a),
        //~^ ERROR cannot find tuple struct or tuple variant `FooB` in this scope
        _ => (),
    }

    match f {
        FooB() => (),
        //~^ ERROR cannot find tuple struct or tuple variant `FooB` in this scope
        _ => (),
    }
}
