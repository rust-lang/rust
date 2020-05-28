#![warn(clippy::needless_bool)]
#![allow(
    unused,
    dead_code,
    clippy::no_effect,
    clippy::if_same_then_else,
    clippy::needless_return
)]

fn main() {
    let x = true;
    let y = false;
    if x {
        true
    } else {
        true
    };
    if x {
        false
    } else {
        false
    };
    if x {
        x
    } else {
        false
    }; // would also be questionable, but we don't catch this yet
    bool_ret(x);
    bool_ret2(x);
}

fn bool_ret(x: bool) -> bool {
    if x {
        return true;
    } else {
        return true;
    };
}

fn bool_ret2(x: bool) -> bool {
    if x {
        return false;
    } else {
        return false;
    };
}
