//@ edition: 2024
#![feature(if_let_guard)]
#![allow(irrefutable_let_patterns)]

fn same_pattern(c: bool) {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, 2) if let y = x && c => (),
        (1, 2) if let z = x => (), //~ ERROR use of moved value: `x`
        _ => (),
    }
}

fn same_pattern_ok(c: bool) {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, 2) if c && let y = x => (),
        (1, 2) if let z = x => (),
        _ => (),
    }
}

fn different_patterns(c: bool) {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, _) if let y = x && c => (),
        (_, 2) if let z = x => (), //~ ERROR use of moved value: `x`
        _ => (),
    }
}

fn different_patterns_ok(c: bool) {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, _) if c && let y = x => (),
        (_, 2) if let z = x => (),
        _ => (),
    }
}

fn or_pattern(c: bool) {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, _) | (_, 2) if let y = x && c => (), //~ ERROR use of moved value: `x`
        _ => (),
    }
}

fn or_pattern_ok(c: bool) {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, _) | (_, 2) if c && let y = x => (),
        _ => (),
    }
}

fn use_in_arm(c: bool) {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, 2) if let y = x && c => false,
        _ => { *x == 1 }, //~ ERROR use of moved value: `x`
    };
}

fn use_in_arm_ok(c: bool) {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, 2) if c && let y = x => false,
        _ => { *x == 1 },
    };
}

fn use_in_same_chain(c: bool) {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, 2) if let y = x && c && let z = x => false, //~ ERROR use of moved value: `x`
        _ => true,
    };
}

fn main() {}
