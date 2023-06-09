#![feature(rustc_attrs, stmt_expr_attributes)]
#![deny(unused_allocation)]

fn main() {
    _ = (#[rustc_box] Box::new([1])).len(); //~ error: unnecessary allocation, use `&` instead
    _ = Box::new([1]).len(); //~ error: unnecessary allocation, use `&` instead
}
