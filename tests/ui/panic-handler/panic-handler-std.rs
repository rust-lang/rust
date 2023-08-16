// normalize-stderr-test "loaded from .*libstd-.*.rlib" -> "loaded from SYSROOT/libstd-*.rlib"
//@error-in-other-file: found duplicate lang item `panic_impl`


use std::panic::PanicInfo;

#[panic_handler]
fn panic(info: PanicInfo) -> ! {
    loop {}
}

fn main() {}
