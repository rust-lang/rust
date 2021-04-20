// run-pass

#![feature(edition_macro_expr)]
#![feature(inline_const)]
#![allow(incomplete_features)]

macro_rules! new_const {
    ($e:expr202x) => {
        $e
    };
    (const $e:block) => {
        1
    };
}

macro_rules! old_const {
    ($e:expr2015) => {
        $e
    };
    (const $e:block) => {
        1
    };
}

fn main() {
    match 1 {
        old_const!(const { 2 }) => (),
        _ => unreachable!(),
    }
    match 1 {
        new_const!(const { 2 }) => unreachable!(),
        _ => (),
    }
}
