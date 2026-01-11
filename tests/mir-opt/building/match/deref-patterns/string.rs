// skip-filecheck
//@ compile-flags: -Z mir-opt-level=0 -C panic=abort

#![feature(deref_patterns)]
#![expect(incomplete_features)]
#![crate_type = "lib"]

// EMIT_MIR string.foo.PreCodegen.after.mir
pub fn foo(s: Option<String>) -> i32 {
    match s {
        Some("a") => 1234,
        s => 4321,
    }
}
