//@ pp-exact

#![feature(decl_macro)]

pub(crate) macro mac { ($arg : expr) => { $arg + $arg } }

fn main() {}
