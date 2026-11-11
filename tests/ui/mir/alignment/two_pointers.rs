//@ run-crash
//@ ignore-i686-pc-windows-msvc: #112480
//@ compile-flags: -C debug-assertions
//@ error-pattern: misaligned pointer dereference: address must be a multiple of 0x4 but is

fn main() {
    let x = [0u32; 2];
    let ptr = x.as_ptr();
    let mut dest = 0u32;
    let dest_ptr = &mut dest as *mut u32;
    unsafe {
        *dest_ptr = *(ptr.byte_add(1));
    }
}
