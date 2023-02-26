#![crate_type = "lib"]

// EMIT_MIR casts.redundant.InstCombine.diff
// EMIT_MIR casts.redundant.PreCodegen.after.mir
pub fn redundant<'a, 'b: 'a>(x: *const &'a u8) -> *const &'a u8 {
    generic_cast::<&'a u8, &'b u8>(x) as *const &'a u8
}

#[inline]
fn generic_cast<T, U>(x: *const T) -> *const U {
    x as *const U
}

// EMIT_MIR casts.roundtrip.PreCodegen.after.mir
pub fn roundtrip(x: *const u8) -> *const u8 {
    x as *mut u8 as *const u8
}
