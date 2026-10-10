// Test that we do not infer the expected types of patterns to an array
// if we're in a refutable pattern.
#![allow(unused_variables)]

struct Zeroes;

impl Into<[usize; 3]> for Zeroes {
    fn into(self) -> [usize; 3] {
        [0; 3]
    }
}

fn let_else() {
    let [a, b, c] = Zeroes.into() else {
        //~^ ERROR type annotations needed
        unreachable!();
    };
}

fn if_let() {
    if let [a, b, c] = Zeroes.into() {
        //~^ ERROR type annotations needed
        unreachable!();
    }
}

fn if_let_else() {
    if let [a, b, c] = Zeroes.into() {
        //~^ ERROR type annotations needed
        unreachable!();
    } else {
        unreachable!();
    }
}

fn main() {}
