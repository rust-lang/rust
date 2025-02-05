#![feature(prelude_import)]
#![no_std]
#![feature(postfix_match)]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;

use std::ops::Add;

//@ pretty-mode:expanded
//@ pp-exact:precedence.pp

macro_rules! repro { ($e:expr) => { $e.match { _ => {} } }; }

struct Struct {}

impl Add<Struct> for usize {
    type Output = ();
    fn add(self, _: Struct) -> () { () }
}
pub fn main() {
    let a;
    (
        { 1 } + 1).match {
        _ => {}
    };
    (4 as usize).match { _ => {} };
    return.match { _ => {} };
    (a = 42).match { _ => {} };
    (|| {}).match { _ => {} };
    (42..101).match { _ => {} };
    (1 + Struct {}).match { _ => {} };
}
