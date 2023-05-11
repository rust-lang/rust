fn main() {
    let input = vec![];
    let _variant_a_result = variant_a(&input);
    let _variant_b_result = variant_b(&input);
}

pub fn variant_a(input: &[(usize, usize, usize, usize)]) -> usize {
    input.iter().filter(|(a, b, c, d)| a <= c && d <= b || c <= a && b <= d).count()
}

pub fn variant_b(input: &[(usize, usize, usize, usize)]) -> usize {
    input.iter().filter(|&&(a, b, c, d)| a <= c && d <= b || c <= a && b <= d).count()
}

// EMIT_MIR slice_filter.variant_a-{closure#0}.ReferencePropagation.diff
// EMIT_MIR slice_filter.variant_a-{closure#0}.CopyProp.diff
// EMIT_MIR slice_filter.variant_a-{closure#0}.DestinationPropagation.diff
// EMIT_MIR slice_filter.variant_b-{closure#0}.CopyProp.diff
// EMIT_MIR slice_filter.variant_b-{closure#0}.ReferencePropagation.diff
// EMIT_MIR slice_filter.variant_b-{closure#0}.DestinationPropagation.diff
