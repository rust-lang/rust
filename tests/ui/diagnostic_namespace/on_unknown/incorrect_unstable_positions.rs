//@ check-pass

#![feature(custom_inner_attributes)]
#![feature(stmt_expr_attributes)]

fn main() {
    #[diagnostic::on_unknown(message = "anonymous block")]
    //~^ WARN cannot be used on
    {}
}
