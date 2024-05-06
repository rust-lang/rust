fn main() {
    println!("cargo::rustc-check-cfg=cfg(assert_no_panic)");
}
