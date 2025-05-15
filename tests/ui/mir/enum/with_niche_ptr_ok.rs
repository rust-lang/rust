//@ run-pass
//@ compile-flags: -C debug-assertions

fn main() {
    let _val = unsafe {
        std::mem::transmute::<*const u64, Option<unsafe extern "C" fn()>>(std::ptr::null())
    };
    let _val = unsafe {
        std::mem::transmute::<*const u64, Option<unsafe extern "C" fn()>>(u64::MAX as *const _)
    };
    let _val = unsafe { std::mem::transmute::<u64, Option<unsafe extern "C" fn()>>(0) };
    let _val = unsafe { std::mem::transmute::<u64, Option<unsafe extern "C" fn()>>(1) };
    let _val = unsafe { std::mem::transmute::<u64, Option<unsafe extern "C" fn()>>(u64::MAX) };
}
