// Matching against float literals should result in a linter error

#![feature(exclusive_range_pattern)]
#![allow(unused)]
#![forbid(illegal_floating_point_literal_pattern)]

fn main() {
    let x = 42.0;
    match x {
        5.0 => {}, //~ ERROR floating-point types cannot be used in patterns
                   //~| WARNING hard error
                   //~| ERROR floating-point types cannot be used in patterns
                   //~| WARNING this was previously accepted by the compiler but is being
        5.0f32 => {}, //~ ERROR floating-point types cannot be used in patterns
                      //~| WARNING hard error
        -5.0 => {}, //~ ERROR floating-point types cannot be used in patterns
                    //~| WARNING hard error
        1.0 .. 33.0 => {}, //~ ERROR floating-point types cannot be used in patterns
                           //~| WARNING hard error
                           //~| ERROR floating-point types cannot be used in patterns
                           //~| WARNING hard error
        39.0 ..= 70.0 => {}, //~ ERROR floating-point types cannot be used in patterns
                             //~| WARNING hard error
                             //~| ERROR floating-point types cannot be used in patterns
                             //~| WARNING hard error
        _ => {},
    };
    let y = 5.0;
    // Same for tuples
    match (x, 5) {
        (3.14, 1) => {}, //~ ERROR floating-point types cannot be used
                         //~| WARNING hard error
        _ => {},
    }
    // Or structs
    struct Foo { x: f32 };
    match (Foo { x }) {
        Foo { x: 2.0 } => {}, //~ ERROR floating-point types cannot be used
                              //~| WARNING hard error
        _ => {},
    }
}
