#![allow(warnings)]

const X: u8 =
    || -> u8 { 5 }()
    //~^ ERROR calls in constants are limited to constant functions
;

fn main() {}
