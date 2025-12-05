#![no_std]

use core::future::Future;

pub struct NeedsDrop;

impl Drop for NeedsDrop {
    fn drop(&mut self) {}
}

#[panic_handler]
fn handler(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

pub unsafe fn oops(x: *const u32) -> u32 {
    *x
}

pub async fn foo(_: NeedsDrop) {
    async fn bar() {}
    bar().await;
}

pub fn poll_foo(x: &mut core::task::Context<'_>) {
    let _g = NeedsDrop;
    let mut p = core::pin::pin!(foo(NeedsDrop));
    let _ = p.as_mut().poll(x);
    let _ = p.as_mut().poll(x);
}

pub fn drop_foo() {
    drop(foo(NeedsDrop));
}
