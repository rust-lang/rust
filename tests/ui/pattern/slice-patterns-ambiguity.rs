#![allow(unused_variables)]

struct Zeroes;

const ARR: [usize; 2] = [0; 2];
const ARR2: [usize; 2] = [2; 2];

impl Into<&'static [usize; 2]> for Zeroes {
    fn into(self) -> &'static [usize; 2] {
        &ARR
    }
}

impl Into<&'static [usize]> for Zeroes {
    fn into(self) -> &'static [usize] {
        &ARR2
    }
}

fn let_decl() {
    let &[a, b] = Zeroes.into();
}

fn let_else() {
    let &[a, b] = Zeroes.into() else {
        //~^ ERROR type annotations needed
        unreachable!();
    };
}

fn if_let() {
    if let &[a, b] = Zeroes.into() {
        //~^ ERROR type annotations needed
        unreachable!();
    }
}

fn if_let_else() {
    if let &[a, b] = Zeroes.into() {
        //~^ ERROR type annotations needed
        unreachable!();
    } else {
        unreachable!();
    }
}

fn main() {}
