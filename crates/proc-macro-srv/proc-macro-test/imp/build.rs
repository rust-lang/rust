//! This teaches cargo about our cfg(rust_analyzer)

fn main() {
    println!("cargo:rustc-check-cfg=cfg(rust_analyzer)");
}
