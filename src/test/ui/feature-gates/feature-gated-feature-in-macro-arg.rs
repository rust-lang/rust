// tests that input to a macro is checked for use of gated features. If this
// test succeeds due to the acceptance of a feature, pick a new feature to
// test. Not ideal, but oh well :(

fn main() {
    let a = &[1, 2, 3];
    println!("{}", {
        extern "rust-intrinsic" { //~ ERROR intrinsics are subject to change
            fn atomic_fence();
        }
        atomic_fence();
        42
    });
}
