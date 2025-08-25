//@ check-pass
//@ proc-macro: makro.rs
//@ edition: 2021

makro::check!();

// checks that a proc-macro doesn't know or parse frontmatters at all and instead treats
// it as normal Rust code.
// see auxiliary/makro.rs for how it is tested.

fn main() {}
