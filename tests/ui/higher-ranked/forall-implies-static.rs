//! Regression test for <https://github.com/rust-lang/rust/issues/26217>.

fn foo<T>() where for<'a> T: 'a {}

fn bar<'a>() {
    foo::<&'a i32>();
    //~^ ERROR lifetime may not live long enough
}

fn main() {
    bar();
}
