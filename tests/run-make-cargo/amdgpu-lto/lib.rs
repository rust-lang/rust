#![feature(abi_gpu_kernel)]
#![no_std]

#[panic_handler]
fn panic_handler(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[unsafe(no_mangle)]
fn foo(a: i32, b: i32) -> i32 {
    a + b
}

#[unsafe(no_mangle)]
extern "gpu-kernel" fn kernel() {}
