fn main() {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let linker_script = format!("{}/linker-{}.ld", manifest_dir, arch);

    // Tell cargo to pass the linker script to the linker..
    println!("cargo:rustc-link-arg=-T{linker_script}");
    // ..and to re-run if it changes.
    println!("cargo:rerun-if-changed={linker_script}");
}
