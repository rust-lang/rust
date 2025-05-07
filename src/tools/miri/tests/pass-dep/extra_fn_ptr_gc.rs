//@ignore-target: windows # No `dlsym` on Windows
//@compile-flags: -Zmiri-permissive-provenance

#[path = "../utils/mod.rs"]
mod utils;

type GetEntropyFn = unsafe extern "C" fn(*mut u8, libc::size_t) -> libc::c_int;

fn main() {
    let name = "getentropy\0";
    let addr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr() as *const _) as usize };
    // If the GC does not account for the extra_fn_ptr entry that this dlsym just added, this GC
    // run will delete our entry for the base addr of the function pointer we will transmute to,
    // and the call through the function pointer will report UB.
    utils::run_provenance_gc();

    let ptr = addr as *mut libc::c_void;
    let func: GetEntropyFn = unsafe { std::mem::transmute(ptr) };
    let dest = &mut [0u8];
    unsafe { func(dest.as_mut_ptr(), dest.len()) };
}
