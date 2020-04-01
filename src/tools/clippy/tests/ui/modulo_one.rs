#![warn(clippy::modulo_one)]
#![allow(clippy::no_effect, clippy::unnecessary_operation)]

static STATIC_ONE: usize = 2 - 1;

fn main() {
    10 % 1;
    10 % 2;

    const ONE: u32 = 1 * 1;

    2 % ONE;
    5 % STATIC_ONE;
}
