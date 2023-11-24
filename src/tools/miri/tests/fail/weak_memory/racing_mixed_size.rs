// We want to control preemption here.
//@compile-flags: -Zmiri-preemption-rate=0

#![feature(core_intrinsics)]

use std::ptr;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::*;
use std::thread::spawn;

fn static_atomic_u32(val: u32) -> &'static AtomicU32 {
    let ret = Box::leak(Box::new(AtomicU32::new(val)));
    ret
}

fn split_u32_ptr(dword: *const u32) -> *const [u16; 2] {
    unsafe { std::mem::transmute::<*const u32, *const [u16; 2]>(dword) }
}

// Wine's SRWLock implementation does this, which is definitely undefined in C++ memory model
// https://github.com/wine-mirror/wine/blob/303f8042f9db508adaca02ef21f8de4992cb9c03/dlls/ntdll/sync.c#L543-L566
// It probably works just fine on x86, but Intel does document this as "don't do it!"
pub fn main() {
    let x = static_atomic_u32(0);
    let j1 = spawn(move || {
        x.store(1, Relaxed);
    });

    let j2 = spawn(move || {
        let x_ptr = x as *const AtomicU32 as *const u32;
        let x_split = split_u32_ptr(x_ptr);
        unsafe {
            let hi = ptr::addr_of!((*x_split)[0]);
            std::intrinsics::atomic_load_relaxed(hi); //~ ERROR: (1) 4-byte atomic store on thread `<unnamed>` and (2) 2-byte atomic load
        }
    });

    j1.join().unwrap();
    j2.join().unwrap();
}
