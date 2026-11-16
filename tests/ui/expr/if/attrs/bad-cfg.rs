#![feature(stmt_expr_attributes)]

fn main() {
    let _ = #[cfg(false)] if true {}; //~ ERROR `cfg` is not supported in this position
}
