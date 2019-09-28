#![feature(core_intrinsics)]
fn main() {
    let x: &'static usize =
        &std::intrinsics::size_of::<i32>(); //~ ERROR temporary value dropped while borrowed
}
