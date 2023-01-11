// The non-crate level cases are in issue-43106-gating-of-builtin-attrs.rs.

#![allow(soft_unstable)]
#![test                    = "4200"]
//~^ ERROR cannot determine resolution for the attribute macro `test`
//~^^ ERROR `test` attribute cannot be used at crate level
fn main() {}
