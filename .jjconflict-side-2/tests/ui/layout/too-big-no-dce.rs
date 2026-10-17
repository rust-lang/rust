//! Ensure that we do not dead-code-eliminate `size_of` calls. That might hide "type too big"
//! errors!
//@ build-fail (test needs codegen)
//@ compile-flags: -O
//@ normalize-stderr: "\d{5}\d*" -> "NUMBER"

//~? ERROR too big for the target

#![crate_type = "lib"]

const PTR_BITS_MINUS_1: usize = std::mem::size_of::<*const ()>() * 8 - 1;

#[unsafe(no_mangle)] // ensure this gets monomorphized
pub fn f() {
    assert_valid_type::<[u32; 1 << PTR_BITS_MINUS_1]>();
}

pub fn assert_valid_type<T>() {
    std::mem::size_of::<T>();
}
