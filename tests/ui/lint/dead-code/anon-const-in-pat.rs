// check-pass
#![feature(inline_const_pat)]
#![allow(incomplete_features)]
#![deny(dead_code)]

const fn one() -> i32 {
    1
}

const fn two() -> i32 {
    2
}

const fn three() -> i32 {
    3
}

fn inline_const() {
    // rust-lang/rust#78171: dead_code lint triggers even though function is used in const pattern
    match 1 {
        const { one() } => {}
        _ => {}
    }
}

fn inline_const_range() {
    match 1 {
        1 ..= const { two() } => {}
        _ => {}
    }
}

struct S<const C: i32>;

fn const_generic_arg() {
    match S::<3> {
        S::<{three()}> => {}
    }
}

fn main() {
    inline_const();
    inline_const_range();
    const_generic_arg();
}
