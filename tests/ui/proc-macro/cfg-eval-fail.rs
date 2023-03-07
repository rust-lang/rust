#![feature(cfg_eval)]
#![feature(stmt_expr_attributes)]

fn main() {
    let _ = #[cfg_eval] #[cfg(FALSE)] 0;
    //~^ ERROR removing an expression is not supported in this position
}
