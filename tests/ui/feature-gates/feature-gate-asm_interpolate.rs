//@ needs-asm-support

fn main() {
    unsafe {
        std::arch::asm!("/* {0} */", interpolate "test");
        //~^ ERROR using the ASM `interpolate` operator is experimental
    }
}
