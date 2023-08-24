#![warn(clippy::integer_division)]

fn main() {
    let two = 2;
    let n = 1 / 2;
    //~^ ERROR: integer division
    let o = 1 / two;
    //~^ ERROR: integer division
    let p = two / 4;
    //~^ ERROR: integer division
    let x = 1. / 2.0;
}
