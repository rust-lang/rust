//! Regression test for <https://github.com/rust-lang/rust/issues/118772>

fn main(_: &i32) { //~ ERROR `main` function has wrong type
    println!("Hello, world!");
}
