//! Regression test for <https://github.com/rust-lang/rust/issues/24227>.
//! Ensure we don't ICE on self type in inherent fn.
//@ check-pass

struct Foo<'a> {
    x: &'a u8
}

impl<'a> Foo<'a> {
    fn foo() {
        let mut tmp: Self;
    }

}

fn main() {}
