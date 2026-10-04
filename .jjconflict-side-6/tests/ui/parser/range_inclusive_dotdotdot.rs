// Make sure that inclusive ranges with `...` syntax don't parse.

use std::ops::RangeToInclusive;

fn return_range_to() -> RangeToInclusive<i32> {
    return ...1; //~ERROR unexpected token: `...`
                 //~^HELP  use `..` for an exclusive range
                 //~^^HELP or `..=` for an inclusive range
}

pub fn main() {
    let x = ...0;    //~ERROR unexpected token: `...`
                     //~^HELP  use `..` for an exclusive range
                     //~^^HELP or `..=` for an inclusive range

    let x = 5...5;   //~ERROR unexpected token: `...`
                     //~^HELP  use `..` for an exclusive range
                     //~^^HELP or `..=` for an inclusive range

    for _ in 0...1 {} //~ERROR unexpected token: `...`
                     //~^HELP  use `..` for an exclusive range
                     //~^^HELP or `..=` for an inclusive range
}
