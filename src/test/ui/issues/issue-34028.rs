#![feature(rustc_attrs)]

macro_rules! m {
    () => { #[cfg(any())] fn f() {} }
}

trait T {}
impl T for () { m!(); }

#[rustc_error]
fn main() {} //~ ERROR compilation successful
