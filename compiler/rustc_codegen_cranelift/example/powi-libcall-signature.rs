// Regression test for a compile failure. Defining `__powisf2` with the wrong signature
// must produce a clear error instead of an ICE when `f32.powi()` lowers to that libcall.

fn main() {
    println!("{}", 2.0f32.powi(4));
}

#[unsafe(no_mangle)]
fn __powisf2() -> f32 {
    let r = 1f32;
    r
}
