//! Test casts for alignment issues

#![feature(rustc_private)]
extern crate libc;

#[warn(clippy::cast_ptr_alignment)]
#[allow(clippy::no_effect, clippy::unnecessary_operation, clippy::cast_lossless)]
fn main() {
    /* These should be warned against */

    // cast to more-strictly-aligned type
    (&1u8 as *const u8) as *const u16;
    (&mut 1u8 as *mut u8) as *mut u16;

    // cast to more-strictly-aligned type, but with the `pointer::cast` function.
    (&1u8 as *const u8).cast::<u16>();
    (&mut 1u8 as *mut u8).cast::<u16>();

    /* These should be ok */

    // not a pointer type
    1u8 as u16;
    // cast to less-strictly-aligned type
    (&1u16 as *const u16) as *const u8;
    (&mut 1u16 as *mut u16) as *mut u8;
    // For c_void, we should trust the user. See #2677
    (&1u32 as *const u32 as *const std::os::raw::c_void) as *const u32;
    (&1u32 as *const u32 as *const libc::c_void) as *const u32;
    // For ZST, we should trust the user. See #4256
    (&1u32 as *const u32 as *const ()) as *const u32;
}
