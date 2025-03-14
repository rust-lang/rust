//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled

#![crate_type = "lib"]

// Test that even though we return a *const u8 not a &[u8] or a NonNull<u8>, LLVM knows that this
// pointer is nonnull.
// CHECK: nonnull ptr @vec_as_ptr
#[no_mangle]
pub fn vec_as_ptr(v: &Vec<u8>) -> *const u8 {
    v.as_ptr()
}

// Test that even though we return a *const u8 not a &[u8] or a NonNull<u8>, LLVM knows that this
// pointer is nonnull.
// CHECK: nonnull ptr @vec_as_mut_ptr
#[no_mangle]
pub fn vec_as_mut_ptr(v: &mut Vec<u8>) -> *mut u8 {
    v.as_mut_ptr()
}
