//! Regression test for issue <https://github.com/rust-lang/rust/issues/51154>
//! Test that anonymous closure types cannot be coerced to a generic type
//! parameter (F: FnMut()) when trying to box them.

fn foo<F: FnMut()>() {
    let _: Box<F> = Box::new(|| ());
    //~^ ERROR mismatched types
}

fn main() {}
