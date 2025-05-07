// tests that input to a macro is checked for use of gated features. If this
// test succeeds due to the acceptance of a feature, pick a new feature to
// test. Not ideal, but oh well :(

fn main() {
    let a = &[1, 2, 3];
    println!("{}", {
        #[rustc_intrinsic] //~ ERROR the `#[rustc_intrinsic]` attribute is used to declare intrinsics as function items
        unsafe fn atomic_fence();

        atomic_fence(); //~ ERROR: is unsafe
        42
    });
}
