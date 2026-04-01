// Checks for the `integer_to_pointer_transmutes` lint with unsized types
//
// Related to https://github.com/rust-lang/rust/issues/145935

//@ check-pass

#![allow(non_camel_case_types)]
#![allow(unused_unsafe)]

#[cfg(target_pointer_width = "64")]
type usizemetadata = i128;

#[cfg(target_pointer_width = "32")]
type usizemetadata = i64;

unsafe fn unsized_type(a: usize) {
    let _ref = unsafe { std::mem::transmute::<usizemetadata, &'static str>(0xff) };
    //~^ WARN transmuting an integer to a pointer
    let _ptr = unsafe { std::mem::transmute::<usizemetadata, *const [u8]>(0xff) };
    //~^ WARN transmuting an integer to a pointer
}

fn main() {}
