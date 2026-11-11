//@ run-pass
//@ compile-flags: -C debug-assertions

fn main() {
    let _val = unsafe {
        std::mem::transmute::<*const usize, Option<unsafe extern "C" fn()>>(std::ptr::null())
    };
    let _val = unsafe {
        std::mem::transmute::<*const usize, Option<unsafe extern "C" fn()>>(usize::MAX as *const _)
    };
    let _val = unsafe { std::mem::transmute::<usize, Option<unsafe extern "C" fn()>>(0) };
    let _val = unsafe { std::mem::transmute::<usize, Option<unsafe extern "C" fn()>>(1) };
    let _val = unsafe { std::mem::transmute::<usize, Option<unsafe extern "C" fn()>>(usize::MAX) };
}
