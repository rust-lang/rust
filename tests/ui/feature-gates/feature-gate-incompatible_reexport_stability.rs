//@ check-pass
//@ normalize-stderr: "(\n)\n$" -> "$1"
// This lint is only available with `staged_api`.
#![allow(incompatible_reexport_stability)]
//~^ WARNING unknown lint: `incompatible_reexport_stability`

fn main() {}
