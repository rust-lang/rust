#![no_std]

extern crate alloc;

use core::alloc::{GlobalAlloc, Layout};
use core::mem::MaybeUninit;

#[global_allocator]
static ALLOC: GlobalDlmalloc = GlobalDlmalloc;

struct GlobalDlmalloc;

unsafe impl GlobalAlloc for GlobalDlmalloc {
    #[inline]
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        core::ptr::null_mut()
    }

    #[inline]
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[used]
static mut BUF: MaybeUninit<[u8; 1024]> = MaybeUninit::uninit();

#[unsafe(no_mangle)]
extern "C" fn init() {
    alloc::alloc::handle_alloc_error(Layout::new::<[u8; 64 * 1024]>());
}

#[panic_handler]
fn my_panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
