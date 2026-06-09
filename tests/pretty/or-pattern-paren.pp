#![feature(prelude_import)]
#![no_std]
#![feature(box_patterns)]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;

//@ pretty-compare-only
//@ pretty-mode:expanded
//@ pp-exact:or-pattern-paren.pp

macro_rules! or_pat { ($($name:pat),+) => { $($name)|+ } }

fn check_at(x: Option<i32>) {
    match x {
        Some(v @ (1 | 2 | 3)) =>


            {
            ::std::io::_print(format_args!("{0}\n", v));
        }
        _ => {}
    }
}
fn check_ref(x: &i32) { match x { &(1 | 2 | 3) => {} _ => {} } }
fn check_box(x: Box<i32>) { match x { box (1 | 2 | 3) => {} _ => {} } }
fn main() { check_at(Some(2)); check_ref(&1); check_box(Box::new(1)); }
