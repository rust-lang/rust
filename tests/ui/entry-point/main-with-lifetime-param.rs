//! Regression test for <https://github.com/rust-lang/rust/issues/51022>

fn main<'a>() {}
//~^ ERROR `main` function is not allowed to have generic parameters [E0131]
