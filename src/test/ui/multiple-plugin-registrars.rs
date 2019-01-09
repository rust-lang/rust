// error-pattern: multiple plugin registration functions found

#![feature(plugin_registrar)]

// the registration function isn't typechecked yet
#[plugin_registrar]
pub fn one() {}

#[plugin_registrar]
pub fn two() {}

fn main() {}
