//@ run-pass
//@ ignore-android no libc
//@ ignore-emscripten no libc
//@ ignore-sgx no libc
//@ only-linux
//@ compile-flags:-C panic=abort
//@ aux-build:helper.rs

#![feature(rustc_private, lang_items, panic_unwind)]
#![feature(alloc_error_handler)]
#![no_std]
#![no_main]

extern crate alloc;
extern crate libc;
extern crate unwind; // For _Unwind_Resume

// ARM targets need these symbols
#[no_mangle]
pub fn __aeabi_unwind_cpp_pr0() {}

#[no_mangle]
pub fn __aeabi_unwind_cpp_pr1() {}

use alloc::boxed::Box;
use alloc::string::ToString;
use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;

extern crate helper;

struct MyAllocator;

#[alloc_error_handler]
fn my_oom(layout: Layout) -> ! {
    use alloc::fmt::write;
    unsafe {
        let size = layout.size();
        let mut s = alloc::string::String::new();
        write(&mut s, format_args!("My OOM: failed to allocate {} bytes!\n", size)).unwrap();
        libc::write(libc::STDERR_FILENO, s.as_ptr() as *const _, s.len());
        libc::exit(0)
    }
}

unsafe impl GlobalAlloc for MyAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() < 4096 { libc::malloc(layout.size()) as _ } else { null_mut() }
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[global_allocator]
static A: MyAllocator = MyAllocator;

#[panic_handler]
fn panic(panic_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        let s = panic_info.to_string();
        const PSTR: &str = "panic occurred: ";
        const CR: &str = "\n";
        libc::write(libc::STDERR_FILENO, PSTR.as_ptr() as *const _, PSTR.len());
        libc::write(libc::STDERR_FILENO, s.as_ptr() as *const _, s.len());
        libc::write(libc::STDERR_FILENO, CR.as_ptr() as *const _, CR.len());
        libc::exit(1)
    }
}

// Because we are compiling this code with `-C panic=abort`, this wouldn't normally be needed.
// However, `core` and `alloc` are both compiled with `-C panic=unwind`, which means that functions
// in these libraries will refer to `rust_eh_personality` if LLVM can not *prove* the contents won't
// unwind. So, for this test case we will define the symbol.
#[lang = "eh_personality"]
extern "C" fn rust_eh_personality(
    _version: i32,
    _actions: i32,
    _exception_class: u64,
    _exception_object: *mut (),
    _context: *mut (),
) -> i32 {
    loop {}
}

#[derive(Default, Debug)]
struct Page(#[allow(dead_code)] [[u64; 32]; 16]);

#[no_mangle]
fn main(_argc: i32, _argv: *const *const u8) -> isize {
    let zero = Box::<Page>::new(Default::default());
    helper::work_with(&zero);
    1
}
