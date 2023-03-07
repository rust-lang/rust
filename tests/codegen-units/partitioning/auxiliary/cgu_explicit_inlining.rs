#![crate_type = "lib"]

#[inline]
pub fn inlined() {}

#[inline(always)]
pub fn always_inlined() {}

#[inline(never)]
pub fn never_inlined() {}
