// check-pass
// aux-build:lint-for-crate-rpass.rs
// ignore-stage1
// compile-flags: -D crate-not-okay

#![feature(plugin, register_attr, custom_inner_attributes)]

#![register_attr(
    crate_okay,
    crate_blue,
    crate_red,
    crate_grey,
    crate_green,
)]

#![plugin(lint_for_crate_rpass)] //~ WARNING compiler plugins are deprecated
#![crate_okay]
#![crate_blue]
#![crate_red]
#![crate_grey]
#![crate_green]

fn main() {}
