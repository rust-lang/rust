#![feature(deref_patterns)]

fn main() {
    let vec![const { vec![] }]: Vec<usize> = vec![];
    //~^ ERROR expected a pattern, found a function call
    //~| ERROR expected a pattern, found a function call
    //~| ERROR expected tuple struct or tuple variant
    //~| ERROR arbitrary expressions aren't allowed in patterns
}
