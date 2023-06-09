// compile-flags: -Zunpretty=hir
// check-pass

#![feature(stmt_expr_attributes, rustc_attrs)]

fn main() {
    let _ = #[rustc_box]
    Box::new(1);
}
