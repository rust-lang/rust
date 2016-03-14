#![feature(plugin)]
#![plugin(clippy)]

#[allow(if_same_then_else)]
#[deny(needless_bool)]
fn main() {
    let x = true;
    if x { true } else { true }; //~ERROR this if-then-else expression will always return true
    if x { false } else { false }; //~ERROR this if-then-else expression will always return false
    if x { true } else { false }; //~ERROR you can reduce this if-then-else expression to just `x`
    if x { false } else { true }; //~ERROR you can reduce this if-then-else expression to just `!x`
    if x { x } else { false }; // would also be questionable, but we don't catch this yet
    bool_ret(x);
    bool_ret2(x);
    bool_ret3(x);
    bool_ret4(x);
}

#[deny(needless_bool)]
#[allow(if_same_then_else)]
fn bool_ret(x: bool) -> bool {
    if x { return true } else { return true }; //~ERROR this if-then-else expression will always return true
}

#[deny(needless_bool)]
#[allow(if_same_then_else)]
fn bool_ret2(x: bool) -> bool {
    if x { return false } else { return false }; //~ERROR this if-then-else expression will always return false
}

#[deny(needless_bool)]
fn bool_ret3(x: bool) -> bool {
    if x { return true } else { return false }; //~ERROR you can reduce this if-then-else expression to just `return x`
}

#[deny(needless_bool)]
fn bool_ret4(x: bool) -> bool {
    if x { return false } else { return true }; //~ERROR you can reduce this if-then-else expression to just `return !x`
}
