#![deny(fall_back_to_never_type)]

macro_rules! unreachable1 {
    () => {{ panic!("internal error: entered unreachable code") }};
}

fn get_unchecked() {
    unreachable1!();
}

fn main() {}
