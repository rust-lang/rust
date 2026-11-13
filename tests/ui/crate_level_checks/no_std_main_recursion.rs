//@check-pass
//@compile-flags: -Cpanic=abort
#![no_std]
#[warn(clippy::main_recursion)]
#[allow(unconditional_recursion)]
fn main() {
    main();
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
