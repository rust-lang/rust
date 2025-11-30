use std::arch::naked_asm;

#[unsafe(naked)]
#[no_mangle]
extern "C" fn used() {
    naked_asm!("ret")
}

#[unsafe(naked)]
#[no_mangle]
extern "C" fn unused() {
    naked_asm!("ret")
}

#[unsafe(naked)]
#[link_section = "foobar"]
#[no_mangle]
extern "C" fn unused_link_section() {
    naked_asm!("ret")
}

fn main() {
    used();
}
