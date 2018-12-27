#![feature(stmt_expr_attributes)]
#![feature(custom_test_frameworks)]

fn main() {
    let _ = #[cfg(unset)] ();
    //~^ ERROR removing an expression is not supported in this position
    let _ = 1 + 2 + #[cfg(unset)] 3;
    //~^ ERROR removing an expression is not supported in this position
    let _ = [1, 2, 3][#[cfg(unset)] 1];
    //~^ ERROR removing an expression is not supported in this position
}
