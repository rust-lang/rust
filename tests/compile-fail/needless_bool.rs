#![feature(plugin)]
#![plugin(clippy)]

#[deny(needless_bool)]
fn main() {
    let x = true;
    if x { true } else { true }; //~ERROR
    if x { false } else { false }; //~ERROR
    if x { true } else { false }; //~ERROR
    if x { false } else { true }; //~ERROR
    if x { x } else { false }; // would also be questionable, but we don't catch this yet
}
