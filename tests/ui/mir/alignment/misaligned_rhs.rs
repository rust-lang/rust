//@ run-crash
//@ ignore-i686-pc-windows-msvc: #112480
//@ compile-flags: -C debug-assertions
//@ error-pattern: misaligned pointer dereference: address must be a multiple of 0x4 but is

fn main() {
    let mut x = [0u32; 2];
    let ptr = x.as_mut_ptr();
    unsafe {
        let _v = *(ptr.byte_add(1));
    }
}
