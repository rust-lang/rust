#![feature(plugin)]
#![plugin(clippy)]

#[deny(needless_bool)]
fn main() {
    let x = true;
    if x { true } else { true }; //~ERROR your if-then-else expression will always return true
    if x { false } else { false }; //~ERROR your if-then-else expression will always return false
    if x { true } else { false }; //~ERROR you can reduce your if statement to its predicate
    if x { false } else { true }; //~ERROR you can reduce your if statement to `!` + its predicate
    if x { x } else { false }; // would also be questionable, but we don't catch this yet
}
