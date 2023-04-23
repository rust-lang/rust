#![feature(alloc_error_handler)]
#![deny(deprecated)]


extern crate alloc;
use alloc::layout::Layout;

#[alloc_error_handler]
fn alloc_error(_l: Layout) -> ! {
    loop {}
}

fn main() {}
