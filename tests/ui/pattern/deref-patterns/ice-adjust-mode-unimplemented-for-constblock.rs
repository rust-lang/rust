#![feature(deref_patterns)]
#![expect(incomplete_features)]

fn main() {
    let vec![const { vec![] }]: Vec<usize> = vec![];
    //~^ ERROR expected a pattern, found a function call
    //~| ERROR usage of qualified paths in this context is experimental
    //~| ERROR expected tuple struct or tuple variant
    //~| ERROR arbitrary expressions aren't allowed in patterns
}
