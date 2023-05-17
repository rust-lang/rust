// compile-flags: -O -C debuginfo=0 -Zmir-opt-level=2
// only-64bit
// ignore-debug

#![crate_type = "lib"]

// EMIT_MIR mem_swap.mem_swap_generic.PreCodegen.after.mir
pub fn mem_swap_generic<T>(a: &mut T, b: &mut T) {
    std::mem::swap(a, b)
}

// EMIT_MIR mem_swap.mem_swap_simple.PreCodegen.after.mir
pub fn mem_swap_simple(a: &mut u32, b: &mut u32) {
    std::mem::swap(a, b)
}

// EMIT_MIR mem_swap.mem_swap_complex.PreCodegen.after.mir
pub fn mem_swap_complex(a: &mut [u16; 13], b: &mut [u16; 13]) {
    std::mem::swap(a, b)
}
