//@ compile-flags: --dap
//@ priroda-relax-exit-status
#![allow(deref_nullptr)]
fn main() {
    unsafe {
        *std::ptr::null_mut::<u8>() = 1;
    }
}
