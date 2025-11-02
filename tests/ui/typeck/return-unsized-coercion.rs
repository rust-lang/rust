// Regression test for issue <https://github.com/rust-lang/rust/issues/134355>

fn digit() -> str {
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
    return { i32::MIN };
}

fn main() {}
