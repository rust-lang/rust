#![allow(unused)]

macro_rules! m { ($i) => {} }
//~^ ERROR missing fragment specifier
//~| WARN previously accepted

fn main() {}
