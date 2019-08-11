#![warn(clippy::out_of_bounds_indexing)]
#![allow(clippy::no_effect, clippy::unnecessary_operation)]

fn main() {
    let empty: [i8; 0] = [];
    empty[0];
    &empty[1..5];
    &empty[0..=4];
    &empty[..=4];
    &empty[1..];
    &empty[..4];
    &empty[0..=0];
    &empty[..=0];

    &empty[0..]; // Ok, should not produce stderr.
    &empty[0..0]; // Ok, should not produce stderr.
    &empty[..0]; // Ok, should not produce stderr.
    &empty[..]; // Ok, should not produce stderr.
}
