// Suppress by using crate

//@ edition:2015
//@ check-pass
//@ aux-crate:bar=bar.rs

#![warn(unused_crate_dependencies)]

extern crate bar;

fn main() {
    println!("bar {}", bar::BAR);
}
