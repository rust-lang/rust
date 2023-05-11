// run-rustfix

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
        //~^ ERROR expected tuple struct or tuple variant, found variant `FooB`
        _ => (),
    }

    match f {
        FooB(x, y) => println!("{} {}", x, y),
        //~^ ERROR expected tuple struct or tuple variant, found variant `FooB`
        _ => (),
    }

    match f {
        FooA(Some(x), y) => println!("{} {}", x, y),
        //~^ ERROR expected tuple struct or tuple variant, found variant `FooA`
        _ => (),
    }

    match f {
        FooB(a, _, _) => println!("{}", a),
        //~^ ERROR expected tuple struct or tuple variant, found variant `FooB`
        _ => (),
    }

    match f {
        FooB() => (),
        //~^ ERROR expected tuple struct or tuple variant, found variant `FooB`
        _ => (),
    }
}
