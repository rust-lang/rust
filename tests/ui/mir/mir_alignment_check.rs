// run-fail
// ignore-wasm32-bare: No panic messages
// ignore-i686-pc-windows-msvc: #112480
// compile-flags: -C debug-assertions
// error-pattern: misaligned pointer dereference: address must be a multiple of 0x4 but is

fn main() {
    let mut x = [0u32; 2];
    let ptr: *mut u8 = x.as_mut_ptr().cast::<u8>();
    unsafe {
        *(ptr.add(1).cast::<u32>()) = 42;
    }
}
