#![feature(stmt_expr_attributes)]
fn okay() -> u32 {
    (
        // Comments in parentheses-expressions caused attributes to be duplicated.
        #[allow(unused_variables)]
        0
    )
}
