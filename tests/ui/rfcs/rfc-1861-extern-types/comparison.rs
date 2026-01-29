// Foreign type tests not covering all operations
//@ only-nightly
//@ build-pass

#![feature(extern_types)]

#![allow(ambiguous_wide_pointer_comparisons)]

extern "C" {
    type ForeignType;
}

#[repr(C)]
struct Example {
    field: ForeignType,
}

fn main() {
    // pointer comparison
    let a = std::ptr::null::<ForeignType>();
    let b = std::ptr::null::<ForeignType>();

    assert!(a == b);

    // field address computation
    let p = std::ptr::null::<Example>();
    unsafe {
        let _ = &(*p).field;
    }

    // pointer casts involving extern types
    let raw = std::ptr::null::<()>();
    let ext = raw as *const ForeignType;
    let _ = ext as *const ();
}
