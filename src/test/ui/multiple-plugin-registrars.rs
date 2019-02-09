// error-pattern: multiple plugin registration functions found

#![feature(plugin_registrar)]

// The registration function isn't type-checked yet.
#[plugin_registrar]
pub fn one() {}

#[plugin_registrar]
pub fn two() {}

fn main() {}
