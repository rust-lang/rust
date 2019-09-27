fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let nightly = std::env::var_os("CARGO_FEATURE_NIGHTLY").is_some();
    let has_stable_alloc = || autocfg::new().probe_rustc_version(1, 36);

    if nightly || has_stable_alloc() {
        autocfg::emit("has_extern_crate_alloc")
    }
}
