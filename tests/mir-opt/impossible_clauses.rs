//@ skip-filecheck
// EMIT_MIR impossible_clauses.impossible_clause.ImpossibleClauses.diff

pub fn impossible_clause(x: &mut i32) -> (&mut i32, &mut i32)
where
    for<'a> &'a mut i32: Copy,
{
    let y = x;
    (y, x)
}
