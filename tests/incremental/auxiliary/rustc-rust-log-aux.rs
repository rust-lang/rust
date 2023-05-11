// rustc-env:RUSTC_LOG=debug
#[cfg(rpass1)]
pub fn foo() {}

#[cfg(rpass2)]
pub fn foo() {
    println!();
}
