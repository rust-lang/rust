// skip-filecheck
//@ compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=2

#![crate_type = "lib"]

pub fn variant_a(input: &[(usize, usize, usize, usize)]) -> usize {
    input.iter().filter(|(a, b, c, d)| a <= c && d <= b || c <= a && b <= d).count()
}

pub fn variant_b(input: &[(usize, usize, usize, usize)]) -> usize {
    input.iter().filter(|&&(a, b, c, d)| a <= c && d <= b || c <= a && b <= d).count()
}

// EMIT_MIR slice_filter.variant_a-{closure#0}.PreCodegen.after.mir
// EMIT_MIR slice_filter.variant_b-{closure#0}.PreCodegen.after.mir
