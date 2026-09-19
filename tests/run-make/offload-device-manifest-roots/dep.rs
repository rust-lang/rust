#![crate_type = "rlib"]

// This tests two things.
//
// First, if we directly compile this for the device, then helper should not end up in the LLVM-IR,
// despite being pub. This makes sure that our mono collector overwrite works.
//
// Second, when we compile host.rs, this file becomes a dependency. In that case its MIR should be
// available, since our Device pass forces `InliningThreshold::Always`.
#[inline(never)]
pub fn helper(x: &mut f32) {
    *x = 1.0;
}
