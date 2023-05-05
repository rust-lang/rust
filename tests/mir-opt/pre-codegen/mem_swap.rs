// compile-flags: -O -C debuginfo=0 -Zmir-opt-level=2 -Z inline-mir-hint-threshold=300
// only-64bit
// ignore-debug

#![crate_type = "lib"]

// EMIT_MIR mem_swap.swap_primitive.PreCodegen.after.mir
pub fn swap_primitive(a: &mut i32, b: &mut i32) {
    std::mem::swap(a, b);
}

// EMIT_MIR mem_swap.swap_generic.PreCodegen.after.mir
pub fn swap_generic<'a, T>(a: &mut T, b: &mut T) {
    std::mem::swap(a, b);
}

// EMIT_MIR mem_swap.swap_big.PreCodegen.after.mir
pub fn swap_big(a: &mut [String; 9], b: &mut [String; 9]) {
    std::mem::swap(a, b);
}
