#![allow(unused_assignments, unused_variables)]

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
    println!("used_only_from_bin_crate_generic_function with {:?}", arg);
}

pub fn used_only_from_this_lib_crate_generic_function<T: Debug>(arg: T) {
    println!("used_only_from_this_lib_crate_generic_function with {:?}", arg);
}

pub fn used_from_bin_crate_and_lib_crate_generic_function<T: Debug>(arg: T) {
    println!("used_from_bin_crate_and_lib_crate_generic_function with {:?}", arg);
}

pub fn used_with_same_type_from_bin_crate_and_lib_crate_generic_function<T: Debug>(arg: T) {
    println!("used_with_same_type_from_bin_crate_and_lib_crate_generic_function with {:?}", arg);
}

pub fn unused_generic_function<T: Debug>(arg: T) {
    println!("unused_generic_function with {:?}", arg);
}

pub fn unused_function() {
    let is_true = std::env::args().len() == 1;
    let mut countdown = 2;
    if !is_true {
        countdown = 20;
    }
}

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

// FIXME(#79651): `used_from_bin_crate_and_lib_crate_generic_function()` is covered and executed
// `2` times, but the coverage output also shows (at the bottom of the coverage report):
//  ------------------
//  | Unexecuted instantiation: <some function name here>
//  ------------------
//
// Note, the function name shown in the error seems to change depending on the structure of the
// code, for some reason, including:
//
// * used_crate::used_from_bin_crate_and_lib_crate_generic_function::<&str>
// * used_crate::use_this_lib_crate
//
// The `Unexecuted instantiation` error may be related to more than one generic function. Since the
// reporting is not consistent, it may not be obvious if there are multiple problems here; however,
// `used_crate::used_from_bin_crate_and_lib_crate_generic_function::<&str>` (which I have seen
// with this error) is the only generic function missing instantiaion coverage counts.
//
// The `&str` variant was called from within this `lib` crate, and the `bin` crate also calls this
// function, but with `T` type `&Vec<i32>`.
//
// I believe the reason is that one or both crates are generating `Zero` counters for what it
// believes are "Unreachable" instantiations, but those instantiations are counted from the
// coverage map in the other crate.
//
// See `add_unreachable_coverage()` in `mapgen.rs` for more on how these `Zero` counters are added
// for what the funciton believes are `DefId`s that did not get codegenned. I suspect the issue
// may be related to this process, but this needs to be confirmed. It may not be possible to know
// for sure if a function is truly unused and should be reported with `Zero` coverage if it may
// still get used from an external crate. (Something to look at: If the `DefId` in MIR corresponds
// _only_ to the generic function without type parameters, is the `DefId` in the codegenned set,
// instantiated with one of the type parameters (in either or both crates) a *different* `DefId`?
// If so, `add_unreachable_coverage()` would assume the MIR `DefId` was uncovered, and would add
// unreachable coverage.
//
// I didn't think they could be different, but if they can, we would need to find the `DefId` for
// the generic function MIR and include it in the set of "codegenned" DefIds if any instantiation
// of that generic function does exist.
//
// Note, however, for `used_with_same_type_from_bin_crate_and_lib_crate_generic_function()` both
// crates use this function with the same type variant. The function does not have multiple
// instantiations, so the coverage analysis is not confused. No "Unexecuted instantiations" errors
// are reported.
