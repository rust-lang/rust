//@ check-pass
//@ normalize-stderr: "(\n)\n$" -> "$1"
// This lint is only available with `staged_api`.
#![allow(ineffective_unstable_reexports)]
//~^ WARNING unknown lint: `ineffective_unstable_reexports`

fn main() {}
