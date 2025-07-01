#![deny(dead_code)] //~ NOTE the lint level is defined here

#[derive(Default)]
struct T; //~ ERROR struct `T` is never constructed

#[derive(Default)]
struct Used;

#[derive(Default)]
enum E { //~ NOTE variant in this enum
    #[default]
    A,
    B, //~ ERROR variant `B` is never constructed
}

// external crate can call T2::default() to construct T2,
// so that no warnings for pub adts
#[derive(Default)]
pub struct T2 {
    _unread: i32,
}

fn main() {
    let _x: Used = Default::default();
    let _e: E = Default::default();
}
