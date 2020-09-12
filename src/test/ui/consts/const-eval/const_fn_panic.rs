// check-pass

#![crate_type = "lib"]
#![feature(const_panic)]

// Can't use assert_{eq, ne}!() yet as panic!() only supports a single argument.

pub const fn always_panic_std() {
    std::panic!("always");
}

pub const fn assert_truth_std() {
    std::assert!(2 + 2 == 4);
}

pub const fn assert_false_std() {
    std::assert!(2 + 2 != 4);
}

pub const fn always_panic_core() {
    core::panic!("always");
}

pub const fn assert_truth_core() {
    core::assert!(2 + 2 == 4);
}

pub const fn assert_false_core() {
    core::assert!(2 + 2 != 4);
}