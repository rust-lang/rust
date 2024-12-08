//@ compile-flags:-C panic=abort

#![no_std]
pub struct Foo;

fn main() {
    Foo.bar()
    //~^ ERROR E0599
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop{}
}
