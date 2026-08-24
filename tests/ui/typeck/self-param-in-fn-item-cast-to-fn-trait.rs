//! Regression test for https://github.com/rust-lang/rust/issues/111411.

pub fn main() {
    fn baz(&self) {}
    //~^ ERROR `self` parameter is only allowed in associated functions
    let _ = &baz as &dyn Fn(i32);
}
