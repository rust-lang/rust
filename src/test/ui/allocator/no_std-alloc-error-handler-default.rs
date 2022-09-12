// run-pass
// ignore-android no libc
// ignore-emscripten no libc
// ignore-sgx no libc
// ignore-wasm32 no libc
// only-linux
// compile-flags:-C panic=abort
// aux-build:helper.rs
// gate-test-default_alloc_error_handler

#![feature(start, rustc_private, new_uninit, panic_info_message, lang_items)]
#![feature(default_alloc_error_handler)]
#![no_std]

extern crate alloc;
extern crate libc;

// ARM targets need these symbols
#[no_mangle]
pub fn __aeabi_unwind_cpp_pr0() {}

#[no_mangle]
pub fn __aeabi_unwind_cpp_pr1() {}

use alloc::boxed::Box;
use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;

extern crate helper;

struct MyAllocator;

unsafe impl GlobalAlloc for MyAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() < 4096 {
            libc::malloc(layout.size()) as _
        } else {
            null_mut()
        }
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[global_allocator]
static A: MyAllocator = MyAllocator;

#[panic_handler]
fn panic(panic_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            const PSTR: &str = "panic occurred: ";
            const CR: &str = "\n";
            libc::write(libc::STDERR_FILENO, PSTR as *const _ as _, PSTR.len());
            libc::write(libc::STDERR_FILENO, s as *const _ as _, s.len());
            libc::write(libc::STDERR_FILENO, CR as *const _ as _, CR.len());
        }
        if let Some(args) = panic_info.message() {
            let mut s = alloc::string::String::new();
            alloc::fmt::write(&mut s, *args).unwrap();
            let s = s.as_str();
            const PSTR: &str = "panic occurred: ";
            const CR: &str = "\n";
            libc::write(libc::STDERR_FILENO, PSTR as *const _ as _, PSTR.len());
            libc::write(libc::STDERR_FILENO, s as *const _ as _, s.len());
            libc::write(libc::STDERR_FILENO, CR as *const _ as _, CR.len());
        } else {
            const PSTR: &str = "panic occurred\n";
            libc::write(libc::STDERR_FILENO, PSTR as *const _ as _, PSTR.len());
        }
        libc::exit(0)
    }
}

// Because we are compiling this code with `-C panic=abort`, this wouldn't normally be needed.
// However, `core` and `alloc` are both compiled with `-C panic=unwind`, which means that functions
// in these libraries will refer to `rust_eh_personality` if LLVM can not *prove* the contents won't
// unwind. So, for this test case we will define the symbol.
#[lang = "eh_personality"]
extern fn rust_eh_personality() {}

#[derive(Debug)]
struct Page(#[allow(unused_tuple_struct_fields)] [[u64; 32]; 16]);

#[start]
pub fn main(_argc: isize, _argv: *const *const u8) -> isize {
    let zero = Box::<Page>::new_zeroed();
    let zero = unsafe { zero.assume_init() };
    helper::work_with(&zero);
    1
}
