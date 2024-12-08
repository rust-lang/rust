#![feature(stmt_expr_attributes)]
#![feature(custom_test_frameworks)]

fn main() {
    let _ = #[cfg(FALSE)] ();
    //~^ ERROR removing an expression is not supported in this position
    let _ = 1 + 2 + #[cfg(FALSE)] 3;
    //~^ ERROR removing an expression is not supported in this position
    let _ = [1, 2, 3][#[cfg(FALSE)] 1];
    //~^ ERROR removing an expression is not supported in this position
}
