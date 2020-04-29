#![deny(unused_assignments)]

struct Foo {
    x: i16
}

const FOO: Foo = Foo { x: 1 };
const BAR: (i16, bool) = (2, false);

fn main() {
    Foo { x: 2 }.x = 3;
    //~^ ERROR unused assignement to temporary
    Foo { x: 2 }.x += 1;
    //~^ ERROR unused assignement to temporary

    (2, 12).0 += 10;
    //~^ ERROR unused assignement to temporary
    (10, false).1 = true;
    //~^ ERROR unused assignement to temporary

    FOO.x = 2;
    //~^ ERROR unused assignement to temporary
    //~| HELP consider introducing a local variable
    FOO.x -= 6;
    //~^ ERROR unused assignement to temporary
    //~| HELP consider introducing a local variable

    BAR.1 = true;
    //~^ ERROR unused assignement to temporary
    //~| HELP consider introducing a local variable
    BAR.0 *= 2;
    //~^ ERROR unused assignement to temporary
    //~| HELP consider introducing a local variable
}
