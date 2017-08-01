#![feature(plugin, custom_attribute, stmt_expr_attributes)]
#![plugin(clippy)]
#![allow(unused_parens)]

fn main() {
    let x: i32 = 42;
    let _ = #[clippy(author)] (x & 0b1111 == 0);  // suggest trailing_zeros
    let _ = x & 0b1_1111 == 0; // suggest trailing_zeros
    let _ = x & 0b1_1010 == 0; // do not lint
}
