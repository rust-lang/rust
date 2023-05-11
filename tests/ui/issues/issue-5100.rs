#![feature(box_patterns)]


enum A { B, C }

fn main() {
    match (true, false) {
        A::B => (),
//~^ ERROR mismatched types
//~| expected `(bool, bool)`, found `A`
//~| expected tuple `(bool, bool)`
//~| found enum `A`
        _ => ()
    }

    match (true, false) {
        (true, false, false) => ()
//~^ ERROR mismatched types
//~| expected a tuple with 2 elements, found one with 3 elements
//~| expected tuple `(bool, bool)`
//~| found tuple `(_, _, _)`
    }

    match (true, false) {
        (true, false, false) => ()
//~^ ERROR mismatched types
//~| expected a tuple with 2 elements, found one with 3 elements
//~| expected tuple `(bool, bool)`
//~| found tuple `(_, _, _)`
    }

    match (true, false) {
        box (true, false) => ()
//~^ ERROR mismatched types
//~| expected tuple `(bool, bool)`
//~| found struct `Box<_>`
    }

    match (true, false) {
        &(true, false) => ()
//~^ ERROR mismatched types
//~| expected `(bool, bool)`, found `&_`
//~| expected tuple `(bool, bool)`
//~| found reference `&_`
    }


    let v = [('a', 'b')   //~ ERROR expected function, found `(char, char)`
             ('c', 'd'),
             ('e', 'f')];

    for &(x,y) in &v {} // should be OK

    // Make sure none of the errors above were fatal
    let x: char = true; //~  ERROR mismatched types
                        //~| expected `char`, found `bool`
}
