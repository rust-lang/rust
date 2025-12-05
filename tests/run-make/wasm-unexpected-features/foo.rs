#![no_core]
#![crate_type = "cdylib"]
#![feature(no_core, lang_items, allocator_internals, rustc_attrs)]
#![needs_allocator]
#![allow(internal_features)]

#[rustc_std_internal_symbol]
unsafe fn __rust_alloc(_size: usize, _align: usize) -> *mut u8 {
    0 as *mut u8
}

unsafe extern "Rust" {
    #[rustc_std_internal_symbol]
    fn __rust_alloc_error_handler(size: usize, align: usize) -> !;
}

#[used]
static mut BUF: [u8; 1024] = [0; 1024];

#[unsafe(no_mangle)]
extern "C" fn init() {
    unsafe {
        __rust_alloc_error_handler(0, 0);
    }
}

mod minicore {
    #[lang = "pointee_sized"]
    pub trait PointeeSized {}

    #[lang = "meta_sized"]
    pub trait MetaSized: PointeeSized {}

    #[lang = "sized"]
    pub trait Sized: MetaSized {}

    #[lang = "copy"]
    pub trait Copy {}
    impl Copy for u8 {}

    #[lang = "drop_in_place"]
    fn drop_in_place<T>(_: *mut T) {}
}
