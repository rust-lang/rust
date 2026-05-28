#![feature(stmt_expr_attributes)]

fn main() {

    // Test that attributes on parens get concatenated
    // in the expected order in the hir folder.

    #[deny(non_snake_case)] #[allow(non_snake_case)] (
        {
            let X = 0;
            let _ = X;
        }
    );

    #[allow(non_snake_case)] #[deny(non_snake_case)] (
        {
            let X = 0; //~ ERROR snake case name
            let _ = X;
        }
    );

}
