// rustc-env: RUST_BACKTRACE=1
// compile-flags: -Zmiri-disable-isolation
// normalize-stderr-test "at .*/(rust[^/]*|checkout)/library/.*" -> "at RUSTLIB/$$FILE:LL:COL"
// normalize-stderr-test "::<.*>" -> ""


fn main() {
    std::panic!("panicking from libstd");
}
