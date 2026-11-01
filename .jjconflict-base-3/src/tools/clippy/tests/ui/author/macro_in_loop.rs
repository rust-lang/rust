//@ check-pass

#![feature(stmt_expr_attributes)]
#![expect(clippy::uninlined_format_args)]

fn main() {
    #[clippy::author]
    for i in 0..1 {
        println!("{}", i);
    }
}
