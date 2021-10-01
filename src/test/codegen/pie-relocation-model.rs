// compile-flags: -C relocation-model=pie
// only-x86_64-unknown-linux-gnu

#![crate_type = "rlib"]

// With PIE we know local functions cannot be interpositioned, we can mark them
// as dso_local.
// CHECK: define dso_local i8 @call_foreign_fn()
#[no_mangle]
pub fn call_foreign_fn() -> u8 {
    unsafe {
        foreign_fn()
    }
}

// External functions are still marked as non-dso_local, since we don't know if the symbol
// is defined in the binary or in the shared library.
// CHECK: declare zeroext i8 @foreign_fn()
extern "C" {fn foreign_fn() -> u8;}

// CHECK: !{i32 7, !"PIC Level", i32 2}
// CHECK: !{i32 7, !"PIE Level", i32 2}
