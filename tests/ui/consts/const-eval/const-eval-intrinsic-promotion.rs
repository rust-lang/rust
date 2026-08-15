#![feature(core_intrinsics)]
fn main() {
    // Test that calls to intrinsics are never promoted
    let x: &'static () =
        &std::intrinsics::cold_path(); //~ ERROR temporary value dropped while borrowed
}
