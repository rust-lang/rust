//@ compile-flags: -C relocation-model=pic -Copt-level=0

#![crate_type = "rlib"]

// CHECK: define i8 @call_foreign_fn()
#[no_mangle]
pub fn call_foreign_fn() -> u8 {
    unsafe { foreign_fn() }
}

// (Allow but do not require `zeroext` here, because it is not worth effort to
// spell out which targets have it and which ones do not; see rust#97800.)

// CHECK: declare{{( zeroext)?}} i8 @foreign_fn()
extern "C" {
    fn foreign_fn() -> u8;
}

// CHECK: !{i32 {{[78]}}, !"PIC Level", i32 2}
