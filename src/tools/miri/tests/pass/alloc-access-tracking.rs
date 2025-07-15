#![no_std]
#![no_main]
//@compile-flags: -Zmiri-track-alloc-id=20 -Zmiri-track-alloc-accesses -Cpanic=abort
//@normalize-stderr-test: "id 20" -> "id $$ALLOC"
//@only-target: linux # alloc IDs differ between OSes (due to extern static allocations)

extern "Rust" {
    fn miri_alloc(size: usize, align: usize) -> *mut u8;
    fn miri_dealloc(ptr: *mut u8, size: usize, align: usize);
}

#[no_mangle]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let ptr = miri_alloc(123, 1);
        *ptr = 42; // Crucially, only a write is printed here, no read!
        assert_eq!(*ptr, 42);
        miri_dealloc(ptr, 123, 1);
    }
    0
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
