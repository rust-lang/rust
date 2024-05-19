#![feature(stmt_expr_attributes)]

fn main() {
    #[clippy::author]
    for i in 0..1 {
        println!("{}", i);
    }
}
