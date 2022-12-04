fn main() {
    let input = vec![];

    // 1761ms on my machine
    let _variant_a_result = variant_a(&input);

    //  656ms on my machine
    let _variant_b_result = variant_b(&input);
}


// EMIT_MIR slice_filter.variant_a-{closure#0}.DestinationPropagation.diff
pub fn variant_a(input: &[(usize, usize, usize, usize)]) -> usize {
    input.iter().filter(|(a, b, c, d)| a <= c && d <= b || c <= a && b <= d).count()
}


// EMIT_MIR slice_filter.variant_b-{closure#0}.DestinationPropagation.diff
pub fn variant_b(input: &[(usize, usize, usize, usize)]) -> usize {
    input.iter().filter(|&&(a, b, c, d)| a <= c && d <= b || c <= a && b <= d).count()
}
