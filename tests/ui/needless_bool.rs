#![feature(plugin)]
#![plugin(clippy)]
#![deny(needless_bool)]

#[allow(if_same_then_else)]
fn main() {
    let x = true;
    let y = false;
    if x { true } else { true };
    if x { false } else { false };
    if x { true } else { false };
    if x { false } else { true };
    if x && y { false } else { true };
    if x { x } else { false }; // would also be questionable, but we don't catch this yet
    bool_ret(x);
    bool_ret2(x);
    bool_ret3(x);
    bool_ret5(x, x);
    bool_ret4(x);
    bool_ret6(x, x);
}

#[allow(if_same_then_else, needless_return)]
fn bool_ret(x: bool) -> bool {
    if x { return true } else { return true };
}

#[allow(if_same_then_else, needless_return)]
fn bool_ret2(x: bool) -> bool {
    if x { return false } else { return false };
}

#[allow(needless_return)]
fn bool_ret3(x: bool) -> bool {
    if x { return true } else { return false };
}

#[allow(needless_return)]
fn bool_ret5(x: bool, y: bool) -> bool {
    if x && y { return true } else { return false };
}

#[allow(needless_return)]
fn bool_ret4(x: bool) -> bool {
    if x { return false } else { return true };
}

#[allow(needless_return)]
fn bool_ret6(x: bool, y: bool) -> bool {
    if x && y { return false } else { return true };
}
