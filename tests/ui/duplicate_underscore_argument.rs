#![feature(plugin)]
#![plugin(clippy)]

#![deny(duplicate_underscore_argument)]
#[allow(dead_code, unused)]

fn join_the_dark_side(darth: i32, _darth: i32) {} //~ERROR `darth` already exists
fn join_the_light_side(knight: i32, _master: i32) {} // the Force is strong with this one

fn main() {
    join_the_dark_side(0, 0);
    join_the_light_side(0, 0);
}