//@ check-pass

#![feature(custom_inner_attributes)]
#![feature(diagnostic_on_unknown)]
#![feature(stmt_expr_attributes)]

fn main() {
    #[diagnostic::on_unknown(message = "anonymous block")]
    //~^ WARN cannot be used on
    {}
}
