//! Test casts for alignment issues

#![feature(core_intrinsics)]
#![warn(clippy::cast_ptr_alignment)]
#![allow(
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::cast_lossless,
    clippy::borrow_as_ptr
)]

fn main() {
    /* These should be warned against */

    // cast to more-strictly-aligned type
    (&1u8 as *const u8) as *const u16;
    //~^ cast_ptr_alignment

    (&mut 1u8 as *mut u8) as *mut u16;
    //~^ cast_ptr_alignment

    // cast to more-strictly-aligned type, but with the `pointer::cast` function.
    (&1u8 as *const u8).cast::<u16>();
    //~^ cast_ptr_alignment

    (&mut 1u8 as *mut u8).cast::<u16>();
    //~^ cast_ptr_alignment

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

    // Issue #2881
    let mut data = [0u8, 0u8];
    unsafe {
        let ptr = &data as *const [u8; 2] as *const u8;
        let _ = (ptr as *const u16).read_unaligned();
        let _ = core::ptr::read_unaligned(ptr as *const u16);
        let _ = core::intrinsics::unaligned_volatile_load(ptr as *const u16);
        let ptr = &mut data as *mut [u8; 2] as *mut u8;
        (ptr as *mut u16).write_unaligned(0);
        core::ptr::write_unaligned(ptr as *mut u16, 0);
        core::intrinsics::unaligned_volatile_store(ptr as *mut u16, 0);
    }
}
