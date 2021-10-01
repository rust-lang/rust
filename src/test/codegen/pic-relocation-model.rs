// compile-flags: -C relocation-model=pic

#![crate_type = "rlib"]

// CHECK: define i8 @call_foreign_fn()
#[no_mangle]
pub fn call_foreign_fn() -> u8 {
    unsafe {
        foreign_fn()
    }
}

// CHECK: declare zeroext i8 @foreign_fn()
extern "C" {fn foreign_fn() -> u8;}

// CHECK: !{i32 7, !"PIC Level", i32 2}
