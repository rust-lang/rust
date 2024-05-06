// TODO: remove this file and deps/libLLVM-18-rust-1.78.0-nightly.so when
// https://github.com/rust-lang/rust/pull/121967 is merged.
fn main() {
    println!("cargo:rerun-if-changed=deps/libLLVM-18-rust-1.78.0-nightly.so");
    println!("cargo:rustc-link-search=deps");
}
