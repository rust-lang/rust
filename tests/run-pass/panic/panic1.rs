// rustc-env: RUST_BACKTRACE=1
// compile-flags: -Zmiri-disable-isolation
// normalize-stderr-test ".*/(rust|checkout)/library/" -> "RUSTLIB/"
// normalize-stderr-test "RUSTLIB/(.*):\d+:\d+ "-> "RUSTLIB/$1:LL:COL "
// normalize-stderr-test "::<.*>" -> ""

fn main() {
    std::panic!("panicking from libstd");
}
