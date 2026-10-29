//! Regression test for https://github.com/rust-lang/rust/issues/14366

fn main() {
    let _x = "test" as &dyn (::std::any::Any);
    //~^ ERROR the size for values of type
}
