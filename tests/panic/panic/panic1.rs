//@rustc-env: RUST_BACKTRACE=1
//@compile-flags: -Zmiri-disable-isolation

fn main() {
    std::panic!("panicking from libstd");
}
