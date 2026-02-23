//@ check-pass

macro_rules! m {
    () => { #[cfg(false)] fn f() {} }
}

trait T {}
impl T for () { m!(); }

fn main() {}
