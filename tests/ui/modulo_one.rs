#![warn(clippy::modulo_one)]
#![allow(clippy::no_effect, clippy::unnecessary_operation)]

fn main() {
    10 % 1;
    10 % 2;
}
