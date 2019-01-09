// error-pattern: duplicate lang item found: `panic_impl`.


use std::panic::PanicInfo;

#[panic_handler]
fn panic(info: PanicInfo) -> ! {
    loop {}
}

fn main() {}
