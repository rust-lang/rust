//@compile-flags: -Zmiri-permissive-provenance

fn main() {
    // Cast a function pointer such that when returning, the return value gets transmuted
    // from raw ptr to reference. This is ABI-compatible, so it's not the call that
    // should fail, but validation should.
    fn f() -> *const i32 {
        0usize as *const i32
    }

    let g: fn() -> &'static i32 = unsafe { std::mem::transmute(f as fn() -> *const i32) };

    let _x = g();
    //~^ ERROR: encountered a null reference
}
