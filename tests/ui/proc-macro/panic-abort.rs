//@ compile-flags: --crate-type proc-macro -Cpanic=abort
//@ force-host
//@ check-pass

//~? WARN building proc macro crate with `panic=abort` may crash the compiler should the proc-macro panic
