// check-pass
// rustc-env:CARGO_PKG_RUST_VERSION=1.28.0

fn main() {
    let _ = std::env::home_dir();
}
