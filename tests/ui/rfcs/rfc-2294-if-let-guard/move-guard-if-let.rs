// Check that borrowck knows that moves in the pattern for if-let guards
// only happen when the pattern is matched.

//@ build-pass
//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

#![allow(irrefutable_let_patterns)]

fn same_pattern() {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, 2) if let y = x => (),
        (1, 2) if let z = x => (),
        _ => (),
    }
}

fn or_pattern() {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, _) | (_, 2) if let y = x => (),
        _ => (),
    }
}

fn main() {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, 2) if let y = x => false,
        _ => { *x == 1 },
    };
}
