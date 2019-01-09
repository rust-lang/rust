#![feature(box_patterns)]
#![feature(box_syntax)]

enum A { B, C }

fn main() {
    match (true, false) {
        A::B => (),
//~^ ERROR mismatched types
//~| expected type `(bool, bool)`
//~| found type `A`
//~| expected tuple, found enum `A`
        _ => ()
    }

    match (true, false) {
        (true, false, false) => ()
//~^ ERROR mismatched types
//~| expected type `(bool, bool)`
//~| found type `(_, _, _)`
//~| expected a tuple with 2 elements, found one with 3 elements
    }

    match (true, false) {
        (true, false, false) => ()
//~^ ERROR mismatched types
//~| expected type `(bool, bool)`
//~| found type `(_, _, _)`
//~| expected a tuple with 2 elements, found one with 3 elements
    }

    match (true, false) {
        box (true, false) => ()
//~^ ERROR mismatched types
//~| expected type `(bool, bool)`
//~| found type `std::boxed::Box<_>`
    }

    match (true, false) {
        &(true, false) => ()
//~^ ERROR mismatched types
//~| expected type `(bool, bool)`
//~| found type `&_`
//~| expected tuple, found reference
    }


    let v = [('a', 'b')   //~ ERROR expected function, found `(char, char)`
             ('c', 'd'),
             ('e', 'f')];

    for &(x,y) in &v {} // should be OK

    // Make sure none of the errors above were fatal
    let x: char = true; //~  ERROR mismatched types
                        //~| expected char, found bool
}
