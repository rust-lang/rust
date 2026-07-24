#![feature(stmt_expr_attributes)]
#![feature(custom_test_frameworks)]

fn main() {
    let _ = 1 + #[cfg(unix)] 2;
    //~^ ERROR removing an expression is not supported in this position
    let _ = 1 + #[cfg(windows)] 2;
    //~^ ERROR removing an expression is not supported in this position

    let _ = 1 + #[cfg(all())] 2;
    //~^ ERROR removing an expression is not supported in this position
    let _ = #[cfg(false)] ();
    //~^ ERROR removing an expression is not supported in this position
    let _ = 1 + 2 + #[cfg(false)] 3;
    //~^ ERROR removing an expression is not supported in this position
    let _ = [1, 2, 3][#[cfg(false)] 1];
    //~^ ERROR removing an expression is not supported in this position
}
