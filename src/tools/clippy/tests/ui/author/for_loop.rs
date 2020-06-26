#![feature(stmt_expr_attributes)]

fn main() {
    #[clippy::author]
    for y in 0..10 {
        let z = y;
    }
}
