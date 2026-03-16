//@ dont-require-annotations: NOTE

#![feature(box_patterns)]

enum A { B, C }

fn main() {
    match (true, false) {
        A::B => (),
//~^ ERROR mismatched types
//~| NOTE expected `(bool, bool)`, found `A`
//~| NOTE expected tuple `(bool, bool)`
//~| NOTE found enum `A`
        _ => ()
    }

    match (true, false) {
        (true, false, false) => ()
//~^ ERROR mismatched types
//~| NOTE expected a tuple with 2 elements, found one with 3 elements
//~| NOTE expected tuple `(bool, bool)`
//~| NOTE found tuple `(_, _, _)`
    }

    match (true, false) {
        (true, false, false) => ()
//~^ ERROR mismatched types
//~| NOTE expected a tuple with 2 elements, found one with 3 elements
//~| NOTE expected tuple `(bool, bool)`
//~| NOTE found tuple `(_, _, _)`
    }

    match (true, false) {
        box (true, false) => ()
//~^ ERROR mismatched types
//~| NOTE expected tuple `(bool, bool)`
//~| NOTE found struct `Box<_>`
    }

    match (true, false) {
        &(true, false) => ()
//~^ ERROR mismatched types
//~| NOTE expected `(bool, bool)`, found `&_`
//~| NOTE expected tuple `(bool, bool)`
//~| NOTE found reference `&_`
    }


    let v = [('a', 'b')   //~ ERROR expected function, found `(char, char)`
             ('c', 'd'),
             ('e', 'f')];

    for &(x,y) in &v {} // should be OK

    // Make sure none of the errors above were fatal
    let x: char = true; //~  ERROR mismatched types
                        //~| NOTE expected `char`, found `bool`
}
