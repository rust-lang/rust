// pp-exact

#![feature(decl_macro)]

macro mac { ($ arg : expr) => { $ arg + $ arg } }

fn main() { }
