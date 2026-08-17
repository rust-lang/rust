//! Regression test for https://github.com/rust-lang/rust/issues/52533
fn foo(_: impl for<'a> FnOnce(&'a u32, &u32) -> &'a u32) {
}

fn main() {
    foo(|a, b| b)
    //~^ ERROR lifetime may not live long enough
}
