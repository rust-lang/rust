#![allow(unused_assignments, unused_variables)]

// compile-flags: -C opt-level=3
// ^^ validates coverage now works with optimizations
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

#[inline(always)]
pub fn used_inline_function() {
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







#[inline(always)]
pub fn used_only_from_bin_crate_generic_function<T: Debug>(arg: T) {
    println!("used_only_from_bin_crate_generic_function with {:?}", arg);
}
// Expect for above function: `Unexecuted instantiation` (see notes in `used_crate.rs`)

#[inline(always)]
pub fn used_only_from_this_lib_crate_generic_function<T: Debug>(arg: T) {
    println!("used_only_from_this_lib_crate_generic_function with {:?}", arg);
}

#[inline(always)]
pub fn used_from_bin_crate_and_lib_crate_generic_function<T: Debug>(arg: T) {
    println!("used_from_bin_crate_and_lib_crate_generic_function with {:?}", arg);
}

#[inline(always)]
pub fn used_with_same_type_from_bin_crate_and_lib_crate_generic_function<T: Debug>(arg: T) {
    println!("used_with_same_type_from_bin_crate_and_lib_crate_generic_function with {:?}", arg);
}

#[inline(always)]
pub fn unused_generic_function<T: Debug>(arg: T) {
    println!("unused_generic_function with {:?}", arg);
}

#[inline(always)]
pub fn unused_function() {
    let is_true = std::env::args().len() == 1;
    let mut countdown = 2;
    if !is_true {
        countdown = 20;
    }
}

#[inline(always)]
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
