//@ run-fail
//@ ignore-i686-pc-windows-msvc: #112480
//@ compile-flags: -C debug-assertions
//@ check-run-results

fn main() {
    let x = [0u32; 2];
    let ptr = x.as_ptr();
    unsafe {
        let _ptr = &(*(ptr.byte_add(1)));
    }
}
