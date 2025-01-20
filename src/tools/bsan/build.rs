fn main() {
    // Don't rebuild bsan when nothing changed.
    println!("cargo:rerun-if-changed=build.rs");
}
