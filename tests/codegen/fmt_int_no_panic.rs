// Trying to check that formatting u8/u32/u64/etc do not panic.
//
// This test does not correctly do so yet.

//@ compile-flags: -O

#![crate_type = "lib"]

// expected to need to write some kind of `impl core::fmt::Write` on a struct like this to avoid
// unrelated panics if `String::write_str` can't make space..
// struct CanAlwaysBeWrittenTo;

use std::fmt::Write;

// CHECK-LABEL: @format_int_doesnt_panic
#[no_mangle]
pub fn format_int_doesnt_panic(s: &mut String) -> std::fmt::Result {
    // CHECK-NOT: panic
    // ... but wait! this will definitely panic if `s.vec.reserve_for_push()` cannot alloc! this
    // shouldn't pass!
    write!(s, "{:x}", 0u8)?;
    write!(s, "{:x}", u8::MAX)?;
    Ok(())
}
