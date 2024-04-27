//@ check-pass

macro_rules! m {
    () => { #[cfg(any())] fn f() {} }
}

trait T {}
impl T for () { m!(); }

fn main() {}
