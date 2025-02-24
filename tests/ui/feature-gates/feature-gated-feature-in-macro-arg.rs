// tests that input to a macro is checked for use of gated features. If this
// test succeeds due to the acceptance of a feature, pick a new feature to
// test. Not ideal, but oh well :(

fn main() {
    let a = &[1, 2, 3];
    println!("{}", {
        extern "rust-intrinsic" { //~ ERROR "rust-intrinsic" ABI is an implementation detail
            fn atomic_fence();
        }
        atomic_fence(); //~ ERROR: is unsafe
        42
    });
}
