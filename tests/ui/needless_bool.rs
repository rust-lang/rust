#![feature(plugin)]
#![plugin(clippy)]
#![deny(needless_bool)]

#[allow(if_same_then_else)]
fn main() {
    let x = true;
    let y = false;
    if x { true } else { true }; //~ERROR this if-then-else expression will always return true
    if x { false } else { false }; //~ERROR this if-then-else expression will always return false
    if x { true } else { false };
    //~^ ERROR this if-then-else expression returns a bool literal
    //~| HELP you can reduce it to
    //~| SUGGESTION x
    if x { false } else { true };
    //~^ ERROR this if-then-else expression returns a bool literal
    //~| HELP you can reduce it to
    //~| SUGGESTION !x
    if x && y { false } else { true };
    //~^ ERROR this if-then-else expression returns a bool literal
    //~| HELP you can reduce it to
    //~| SUGGESTION !(x && y)
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
    //~^ ERROR this if-then-else expression will always return true
}

#[allow(if_same_then_else, needless_return)]
fn bool_ret2(x: bool) -> bool {
    if x { return false } else { return false };
    //~^ ERROR this if-then-else expression will always return false
}

#[allow(needless_return)]
fn bool_ret3(x: bool) -> bool {
    if x { return true } else { return false };
    //~^ ERROR this if-then-else expression returns a bool literal
    //~| HELP you can reduce it to
    //~| SUGGESTION return x
}

#[allow(needless_return)]
fn bool_ret5(x: bool, y: bool) -> bool {
    if x && y { return true } else { return false };
    //~^ ERROR this if-then-else expression returns a bool literal
    //~| HELP you can reduce it to
    //~| SUGGESTION return x && y
}

#[allow(needless_return)]
fn bool_ret4(x: bool) -> bool {
    if x { return false } else { return true };
    //~^ ERROR this if-then-else expression returns a bool literal
    //~| HELP you can reduce it to
    //~| SUGGESTION return !x
}

#[allow(needless_return)]
fn bool_ret6(x: bool, y: bool) -> bool {
    if x && y { return false } else { return true };
    //~^ ERROR this if-then-else expression returns a bool literal
    //~| HELP you can reduce it to
    //~| SUGGESTION return !(x && y)
}
