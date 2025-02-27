#![warn(clippy::duplicate_underscore_argument)]

fn join_the_dark_side(darth: i32, _darth: i32) {}
//~^ duplicate_underscore_argument

fn join_the_light_side(knight: i32, _master: i32) {} // the Force is strong with this one

fn main() {
    join_the_dark_side(0, 0);
    join_the_light_side(0, 0);
}
