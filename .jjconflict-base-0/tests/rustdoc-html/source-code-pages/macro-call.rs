// This is yet another test to ensure that only macro calls are considered as such
// by the rustdoc highlighter.
// This is a regression test for <https://github.com/rust-lang/rust/issues/151904>.

#![crate_name = "foo"]

//@ has src/foo/macro-call.rs.html
//@ count - '//code/span[@class="macro"]' 2
//@ has - '//code/span[@class="macro"]' 'panic!'
//@ has - '//code/span[@class="macro"]' 'macro_rules!'

pub struct Layout;

impl Layout {
    pub fn new<X: std::fmt::Debug>() {}
}

pub fn bar() {
    let layout = Layout::new::<u32>();
    if layout != Layout::new::<u32>() {
        panic!();
    }
    let macro_rules = 3;
    if macro_rules != 3 {}
}

macro_rules! blob {
    () => {}
}
