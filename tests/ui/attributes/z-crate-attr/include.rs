// `include_str!` and `include!` inside a crate attribute resolve paths relative to the crate root,
// as if the attribute were written at its top.

//@ check-pass
//@ compile-flags: -Zcrate-attr=doc=include_str!("auxiliary/include.rs")
//@ compile-flags: -Zcrate-attr=doc=include!("auxiliary/include.rs")

fn main() {}
