#![allow(unused_assignments, unused_variables)]
// Verify that coverage works with optimizations:
//@ compile-flags: -C opt-level=3

use std::fmt::Debug;

pub fn used_function() {
    // Initialize test constants in a way that cannot be determined at compile time, to ensure
    // rustc and LLVM cannot optimize out statements (or coverage counters) downstream from
    // dependent conditions.
    let is_true = std::env::args().len() == 1;
    let mut countdown = 0;
    if is_true {
        countdown = 10;
    }
    use_this_lib_crate();
}

pub fn used_only_from_bin_crate_generic_function<T: Debug>(arg: T) {
    println!("used_only_from_bin_crate_generic_function with {arg:?}");
}
// Expect for above function: `Unexecuted instantiation` (see below)
pub fn used_only_from_this_lib_crate_generic_function<T: Debug>(arg: T) {
    println!("used_only_from_this_lib_crate_generic_function with {arg:?}");
}

pub fn used_from_bin_crate_and_lib_crate_generic_function<T: Debug>(arg: T) {
    println!("used_from_bin_crate_and_lib_crate_generic_function with {arg:?}");
}

pub fn used_with_same_type_from_bin_crate_and_lib_crate_generic_function<T: Debug>(arg: T) {
    println!("used_with_same_type_from_bin_crate_and_lib_crate_generic_function with {arg:?}");
}

pub fn unused_generic_function<T: Debug>(arg: T) {
    println!("unused_generic_function with {arg:?}");
}

pub fn unused_function() {
    let is_true = std::env::args().len() == 1;
    let mut countdown = 2;
    if !is_true {
        countdown = 20;
    }
}

#[allow(dead_code)]
fn unused_private_function() {
    let is_true = std::env::args().len() == 1;
    let mut countdown = 2;
    if !is_true {
        countdown = 20;
    }
}

fn use_this_lib_crate() {
    used_from_bin_crate_and_lib_crate_generic_function("used from library used_crate.rs");
    used_with_same_type_from_bin_crate_and_lib_crate_generic_function(
        "used from library used_crate.rs",
    );
    let some_vec = vec![5, 6, 7, 8];
    used_only_from_this_lib_crate_generic_function(some_vec);
    used_only_from_this_lib_crate_generic_function("used ONLY from library used_crate.rs");
}

// FIXME(#79651): "Unexecuted instantiation" errors appear in coverage results,
// for example:
//
// | Unexecuted instantiation: used_crate::used_only_from_bin_crate_generic_function::<_>
//
// These notices appear when `llvm-cov` shows instantiations. This may be a
// default option, but it can be suppressed with:
//
// ```shell
// $ `llvm-cov show --show-instantiations=0 ...`
// ```
//
// The notice is triggered because the function is unused by the library itself,
// so when the library is compiled, an "unused" set of mappings for that function
// is included in the library's coverage metadata.
//
// Even though this function is used by `uses_crate.rs` (and
// counted), with substitutions for `T`, those instantiations are only generated
// when the generic function is actually used (from the binary, not from this
// library crate). So the test result shows coverage for all instantiated
// versions and their generic type substitutions, plus the `Unexecuted
// instantiation` message for the non-substituted version. This is valid, but
// unfortunately a little confusing.
//
// The library crate has its own coverage map, and the only way to show unused
// coverage of a generic function is to include the generic function in the
// coverage map, marked as an "unused function". If the library were used by
// another binary that never used this generic function, then it would be valid
// to show the unused generic, with unknown substitution (`_`).
//
// The alternative would be to exclude all generics from being included in the
// "unused functions" list, which would then omit coverage results for
// `unused_generic_function<T>()`.
