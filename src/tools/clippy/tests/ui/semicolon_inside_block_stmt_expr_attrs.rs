// Test when the feature `stmt_expr_attributes` is enabled

#![feature(stmt_expr_attributes)]
#![allow(clippy::no_effect)]
#![warn(clippy::semicolon_inside_block)]

pub fn issue15388() {
    #[rustfmt::skip]
    {0; 0};
    //~^ semicolon_inside_block
}
