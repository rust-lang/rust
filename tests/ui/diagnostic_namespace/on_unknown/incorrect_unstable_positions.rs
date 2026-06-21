//@ check-pass

#![feature(custom_inner_attributes)]
#![feature(diagnostic_on_unknown)]
#![feature(stmt_expr_attributes)]

fn main() {
    #[diagnostic::on_unknown(message = "anonymous block")]
    //~^ WARN `#[diagnostic::on_unknown]` can only be applied to `use` statements and module declarations
    {}
}
