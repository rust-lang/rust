fn main() {
    let f: fn() -> ! = || std::process::exit(0);
    f();

    // FIXME: Also add a test for <https://github.com/rust-lang/rust/issues/66738>, once that is fixed.
}
