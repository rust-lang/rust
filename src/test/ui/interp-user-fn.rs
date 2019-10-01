#![feature(stmt_expr_attributes)]

fn main() {
    let user_body = #[rustc_interp_user_fn] || {};
    //~^ ERROR `#[rustc_interp_user_fn]` is for use by interpreters only [E0658]
    user_body();
}
