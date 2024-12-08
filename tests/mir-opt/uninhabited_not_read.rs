// skip-filecheck

//@ edition: 2021
// In ed 2021 and below, we don't fallback `!` to `()`.
// This would introduce a `! -> ()` coercion which would
// be UB if we didn't disallow this explicitly.

#![feature(never_type)]

// EMIT_MIR uninhabited_not_read.main.SimplifyLocals-final.after.mir
fn main() {
    // With a type annotation
    unsafe {
        let x = 3u8;
        let x: *const ! = &x as *const u8 as *const _;
        let _: ! = *x;
    }

    // Without a type annotation, make sure we don't implicitly coerce `!` to `()`
    // when we do the noop `*x`.
    unsafe {
        let x = 3u8;
        let x: *const ! = &x as *const u8 as *const _;
        let _ = *x;
    }
}
