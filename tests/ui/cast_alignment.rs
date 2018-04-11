//! Test casts for alignment issues

#[warn(cast_ptr_alignment)]
#[allow(no_effect, unnecessary_operation, cast_lossless)]
fn main() {
    /* These should be warned against */

    // cast to more-strictly-aligned type
    (&1u8 as *const u8) as *const u16;
    (&mut 1u8 as *mut u8) as *mut u16;

    /* These should be okay */

    // not a pointer type
    1u8 as u16;
    // cast to less-strictly-aligned type
    (&1u16 as *const u16) as *const u8;
    (&mut 1u16 as *mut u16) as *mut u8;
}
