// Test when the feature `stmt_expr_attributes` is enabled

#![feature(stmt_expr_attributes)]
#![warn(clippy::semicolon_inside_block)]
#![expect(clippy::no_effect)]

pub fn issue15388() {
    #[rustfmt::skip]
    {0; 0};
    //~^ semicolon_inside_block
}
