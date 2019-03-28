// This does nothing.
// It is only there because cargo will only set the $OUT_DIR env variable
// for tests if there is a build script.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
