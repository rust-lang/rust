//@aux-build:proc_macros.rs
#![allow(irrefutable_let_patterns, unused)]
#![warn(clippy::redundant_at_rest_pattern)]

#[macro_use]
extern crate proc_macros;

fn main() {
    if let [a @ ..] = [()] {}
    //~^ redundant_at_rest_pattern
    if let [ref a @ ..] = [()] {}
    //~^ redundant_at_rest_pattern
    if let [mut a @ ..] = [()] {}
    //~^ redundant_at_rest_pattern
    if let [ref mut a @ ..] = [()] {}
    //~^ redundant_at_rest_pattern
    let v = vec![()];
    if let [a @ ..] = &*v {}
    //~^ redundant_at_rest_pattern
    let s = &[()];
    if let [a @ ..] = s {}
    //~^ redundant_at_rest_pattern
    // Don't lint
    if let [..] = &*v {}
    if let [a] = &*v {}
    if let [()] = &*v {}
    if let [first, rest @ ..] = &*v {}
    if let a = [()] {}
    external! {
        if let [a @ ..] = [()] {}
    }
}
