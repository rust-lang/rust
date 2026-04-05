// EMIT_MIR impossible_predicates.impossible_predicate.ImpossiblePredicates.diff

pub fn impossible_predicate(x: &mut i32) -> (&mut i32, &mut i32)
where
    for<'a> &'a mut i32: Copy,
{
    // CHECK-LABEL: fn impossible_predicate(
    // CHECK: bb0: {
    // CHECK-NEXT: unreachable;
    // CHECK-NEXT: }
    let y = x;
    (y, x)
}
