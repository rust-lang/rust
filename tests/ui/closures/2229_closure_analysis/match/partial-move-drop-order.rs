//@ edition:2021
//@ run-pass
//@ check-run-results
//@ aux-build:partial_move_drop_order_lib.rs

extern crate partial_move_drop_order_lib;
use partial_move_drop_order_lib::{LoudDrop, ExtNonExhaustive};

pub enum OneVariant {
    One(i32, LoudDrop),
}

pub enum TwoVariants {
    One(i32, LoudDrop),
    Two,
}

#[non_exhaustive]
pub enum NonExhaustive {
    One(i32, LoudDrop),
}

#[allow(unused)]
fn one_variant() {
    println!("one variant:");
    let mut thing = OneVariant::One(0, LoudDrop("a"));
    let closure = move || match thing {
        OneVariant::One(x, _) => {}
        _ => unreachable!(),
    };
    println!("before assign");
    thing = OneVariant::One(1, LoudDrop("b"));
    println!("after assign");
}

#[allow(unused)]
fn two_variants() {
    println!("two variants:");
    let mut thing = TwoVariants::One(0, LoudDrop("a"));
    let closure = move || match thing {
        TwoVariants::One(x, _) => {}
        _ => unreachable!(),
    };
    println!("before assign");
    thing = TwoVariants::One(1, LoudDrop("b"));
    println!("after assign");
}

#[allow(unused)]
fn non_exhaustive() {
    println!("non exhaustive:");
    let mut thing = NonExhaustive::One(0, LoudDrop("a"));
    let closure = move || match thing {
        NonExhaustive::One(x, _) => {}
        _ => unreachable!(),
    };
    println!("before assign");
    thing = NonExhaustive::One(1, LoudDrop("b"));
    println!("after assign");
}

#[allow(unused)]
fn ext_non_exhaustive() {
    println!("external non exhaustive:");
    let mut thing = ExtNonExhaustive::One(0, LoudDrop("a"));
    let closure = move || match thing {
        ExtNonExhaustive::One(x, _) => {}
        _ => unreachable!(),
    };
    println!("before assign");
    thing = ExtNonExhaustive::One(1, LoudDrop("b"));
    println!("after assign");
}

fn main() {
    one_variant();
    println!();
    two_variants();
    println!();
    non_exhaustive();
    println!();
    ext_non_exhaustive();
}
