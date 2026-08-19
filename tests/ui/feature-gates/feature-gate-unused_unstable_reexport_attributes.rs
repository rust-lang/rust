//@ check-pass
//@ normalize-stderr: "(\n)\n$" -> "$1"
// This lint is only available with `staged_api`.
#![allow(unused_unstable_reexport_attributes)]
//~^ WARNING unknown lint: `unused_unstable_reexport_attributes`

fn main() {}
