// `include_str!` and `include!` inside a crate attribute resolve paths relative to the crate root,
// as if the attribute were written at its top.

//@ compile-flags: -Zcrate-attr=doc=include_str!("auxiliary/include.rs")
//@ compile-flags: -Zcrate-attr=doc=include!("auxiliary/include.rs")

fn main() {}

//~? ERROR cannot resolve relative path in non-file source `<crate attribute>`
//~? ERROR cannot resolve relative path in non-file source `<crate attribute>`
