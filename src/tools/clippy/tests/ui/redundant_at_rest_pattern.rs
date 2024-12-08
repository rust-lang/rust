//@aux-build:proc_macros.rs
#![allow(irrefutable_let_patterns, unused)]
#![warn(clippy::redundant_at_rest_pattern)]

#[macro_use]
extern crate proc_macros;

fn main() {
    if let [a @ ..] = [()] {}
    if let [ref a @ ..] = [()] {}
    if let [mut a @ ..] = [()] {}
    if let [ref mut a @ ..] = [()] {}
    let v = vec![()];
    if let [a @ ..] = &*v {}
    let s = &[()];
    if let [a @ ..] = s {}
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
