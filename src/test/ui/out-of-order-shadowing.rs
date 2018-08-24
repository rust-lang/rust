// aux-build:define_macro.rs
// error-pattern: `bar` is already in scope

macro_rules! bar { () => {} }
define_macro!(bar);
bar!();

macro_rules! m { () => { #[macro_use] extern crate define_macro; } }
m!();

fn main() {}
