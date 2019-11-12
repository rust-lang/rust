// run-pass
// aux-build:lint-for-crate-rpass.rs
// ignore-stage1
// compile-flags: -D crate-not-okay

#![feature(plugin, register_attr, custom_inner_attributes, rustc_attrs)]

#![register_attr(
    rustc_crate_okay,
    rustc_crate_blue,
    rustc_crate_red,
    rustc_crate_grey,
    rustc_crate_green,
)]

#![plugin(lint_for_crate_rpass)] //~ WARNING compiler plugins are deprecated
#![rustc_crate_okay]
#![rustc_crate_blue]
#![rustc_crate_red]
#![rustc_crate_grey]
#![rustc_crate_green]

fn main() {}
