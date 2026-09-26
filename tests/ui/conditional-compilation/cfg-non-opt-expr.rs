#![feature(stmt_expr_attributes)]
#![feature(custom_test_frameworks)]

#[derive(Clone)]
enum E {
    V1 = #[cfg(true)] 1,
    //~^ ERROR removing an expression is not supported in this position
    //~| ERROR removing an expression is not supported in this position
    V2 = #[cfg_attr(true, cfg(true))] 2,
    //~^ ERROR removing an expression is not supported in this position
}

macro_rules! mac {
    ($expr:expr) => { $expr.clone() }
}

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
    let _ = mac!(#[cfg(true)] 10);
    //~^ ERROR removing an expression is not supported in this position
    let _ = #[cfg(true)] ();
    //~^ ERROR removing an expression is not supported in this position
    let _ = 1 + 2 + #[cfg(true)] 3;
    //~^ ERROR removing an expression is not supported in this position
    let _ = [1, 2, 3][#[cfg(true)] 1];
    //~^ ERROR removing an expression is not supported in this position
}
