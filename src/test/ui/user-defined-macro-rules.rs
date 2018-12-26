#![allow(unused_macros)]

macro_rules! macro_rules { () => {} } //~ ERROR user-defined macros may not be named `macro_rules`

fn main() {}
