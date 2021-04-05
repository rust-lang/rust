// AST-based macro attributes expanding to an empty expression produce an error and not ICE.

#![feature(custom_test_frameworks)]
#![feature(stmt_expr_attributes)]
#![feature(test)]

fn main() {
    let _ = #[test] 0; //~ ERROR removing an expression is not supported in this position
    let _ = #[bench] 1; //~ ERROR removing an expression is not supported in this position
    let _ = #[test_case] 2; //~ ERROR removing an expression is not supported in this position
}
