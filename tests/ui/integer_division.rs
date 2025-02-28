#![warn(clippy::integer_division)]

fn main() {
    let two = 2;
    let n = 1 / 2;
    //~^ integer_division

    let o = 1 / two;
    //~^ integer_division

    let p = two / 4;
    //~^ integer_division

    let x = 1. / 2.0;
}
