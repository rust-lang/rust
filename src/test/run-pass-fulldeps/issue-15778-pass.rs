// aux-build:lint-for-crate.rs
// ignore-stage1
// compile-flags: -D crate-not-okay

#![feature(plugin, custom_attribute, custom_inner_attributes, rustc_attrs)]

#![plugin(lint_for_crate)]
#![rustc_crate_okay]
#![rustc_crate_blue]
#![rustc_crate_red]
#![rustc_crate_grey]
#![rustc_crate_green]

fn main() {}
