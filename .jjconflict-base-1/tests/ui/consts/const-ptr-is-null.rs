use std::ptr;

const IS_NULL: () = {
    assert!(ptr::null::<u8>().is_null());
};
const IS_NOT_NULL: () = {
    assert!(!ptr::null::<u8>().wrapping_add(1).is_null());
};

const MAYBE_NULL: () = {
    let x = 15;
    let ptr = &x as *const i32;
    // This one is still unambiguous...
    assert!(!ptr.is_null());
    // and in fact, any offset not visible by 4 (the alignment) cannot be null,
    // even if it goes out-of-bounds...
    assert!(!ptr.wrapping_byte_add(13).is_null());
    assert!(!ptr.wrapping_byte_add(18).is_null());
    assert!(!ptr.wrapping_byte_sub(1).is_null());
    // ... but once we shift outside the allocation, with an offset divisible by 4,
    // we might become null.
    assert!(!ptr.wrapping_sub(512).is_null()); //~ ERROR null-ness of this pointer cannot be determined
};

fn main() {}
