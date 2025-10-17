use std::arch::naked_asm;

#[unsafe(naked)]
#[unsafe(no_mangle)]
extern "C" fn used() {
    naked_asm!("ret")
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
extern "C" fn unused() {
    naked_asm!("ret")
}

#[unsafe(naked)]
#[unsafe(link_section = "foobar")]
#[unsafe(no_mangle)]
extern "C" fn unused_link_section() {
    naked_asm!("ret")
}

fn main() {
    used();
}
