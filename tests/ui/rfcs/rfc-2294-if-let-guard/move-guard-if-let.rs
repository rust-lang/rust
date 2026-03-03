// Check that borrowck doesn't know that moves in the pattern for if-let guards
// only happen when the pattern is matched.
// See <https://github.com/rust-lang/rust/issues/153263>.

#![allow(irrefutable_let_patterns)]

fn same_pattern() {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, 2) if let y = x => (),
        (1, 2) if let z = x => (), //~ ERROR use of moved value: `x`
        _ => (),
    }
}

fn or_pattern() {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, _) | (_, 2) if let y = x => (), //~ ERROR use of moved value: `x`
        _ => (),
    }
}

fn main() {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, 2) if let y = x => false,
        _ => *x == 1, //~ ERROR use of moved value: `x`
    };
}
