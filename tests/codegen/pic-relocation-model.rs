//@ compile-flags: -C relocation-model=pic -Copt-level=0

#![crate_type = "rlib"]

// CHECK: define i8 @call_foreign_fn()
#[no_mangle]
pub fn call_foreign_fn() -> u8 {
    unsafe { foreign_fn() }
}

// CHECK: declare void @foreign_fn()
extern "C" {
    fn foreign_fn() -> u8;
}

// CHECK: !{i32 {{[78]}}, !"PIC Level", i32 2}
