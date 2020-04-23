fn main() {
    // Don't rebuild miri when nothing changed.
    println!("cargo:rerun-if-changed=build.rs");
    // vergen
    vergen::generate_cargo_keys(vergen::ConstantsFlags::all())
        .expect("Unable to generate vergen keys!");
}
