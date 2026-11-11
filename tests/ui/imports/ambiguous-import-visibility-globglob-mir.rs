// issue: rust-lang/rust#159038
// Downstream missing optimized MIR when a restricted glob wins the slot.
// Dead code: `ambiguous-import-visibility-globglob-reachable.rs`.

//@ build-pass
//@ aux-build:ambiguous-import-visibility-globglob-mir.rs

extern crate ambiguous_import_visibility_globglob_mir as dep;

pub fn call_f() -> u32 {
    dep::f()
}

fn main() {
    call_f();
}
