#![warn(clippy::duplicate_underscore_argument)]
#[allow(dead_code, unused)]

fn join_the_dark_side(darth: i32, _darth: i32) {}
//~^ ERROR: `darth` already exists, having another argument having almost the same name ma
//~| NOTE: `-D clippy::duplicate-underscore-argument` implied by `-D warnings`
fn join_the_light_side(knight: i32, _master: i32) {} // the Force is strong with this one

fn main() {
    join_the_dark_side(0, 0);
    join_the_light_side(0, 0);
}
