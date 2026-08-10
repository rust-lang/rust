//@ check-pass
//@ compile-flags: -Zassumptions-on-binders -Znext-solver=no

#![crate_type = "lib"]

//~? WARN unconditionally enables the next trait solver
