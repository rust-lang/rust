//! Regression test for https://github.com/rust-lang/rust/issues/12863

mod foo { pub fn bar() {} }

fn main() {
    match () {
        foo::bar => {}
        //~^ ERROR expected unit struct, unit variant or constant, found function `foo::bar`
    }
}
