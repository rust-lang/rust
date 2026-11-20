// This is a test with a setup similar to issue 85155, which triggers a const eval error: a const
// argument value is outside the range expected by the `stdarch` intrinsic.
//
// It's not the exact code mentioned in that issue because it depends both on `stdarch` intrinsics
// only available on x64, and internal implementation details of `stdarch`. But mostly because these
// are not important to trigger the diagnostics issue: it's specifically about the lack of context
// in the diagnostics of post-monomorphization errors (PMEs) for consts, happening in a dependency.
// Therefore, its setup is reproduced with an aux crate, which will similarly trigger a PME
// depending on the const argument value, like the `stdarch` intrinsics would.
//
//@ aux-build: post_monomorphization_error.rs
//@ build-fail: this is a post-monomorphization error, it passes check runs and requires building
//             to actually fail.

extern crate post_monomorphization_error;

fn main() {
    // This function triggers a PME whenever the const argument does not fit in 1-bit.
    post_monomorphization_error::stdarch_intrinsic::<2>();
    //~^ NOTE the above error was encountered while instantiating
}

//~? ERROR attempt to divide `1_usize` by zero
//~? NOTE erroneous constant encountered
