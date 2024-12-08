//@ run-rustfix
// Check the `unused_parens` suggestion for paren_expr with attributes.
// The suggestion should retain attributes in the front.

#![feature(stmt_expr_attributes)]
#![deny(unused_parens)]

pub fn foo() -> impl Fn() {
    let _ = (#[inline] #[allow(dead_code)] || println!("Hello!")); //~ERROR unnecessary parentheses
    (#[inline] #[allow(dead_code)] || println!("Hello!")) //~ERROR unnecessary parentheses
}

fn main() {
    let _ = foo();
}
