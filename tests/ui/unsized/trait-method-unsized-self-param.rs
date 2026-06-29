//! Regression test for <https://github.com/rust-lang/rust/issues/27078>.

trait Foo {
    const BAR: i32;
    fn foo(self) -> &'static i32 {
        //~^ ERROR the size for values of type
        &<Self>::BAR
    }
}

fn main() {}
