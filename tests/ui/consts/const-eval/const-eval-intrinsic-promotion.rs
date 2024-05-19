#![feature(core_intrinsics)]
fn main() {
    // Test that calls to intrinsics are never promoted
    let x: &'static usize =
        &std::intrinsics::size_of::<i32>(); //~ ERROR temporary value dropped while borrowed
}
