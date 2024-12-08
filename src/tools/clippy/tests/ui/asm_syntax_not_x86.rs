//@ignore-target: i686 x86
//@needs-asm-support

#[warn(clippy::inline_asm_x86_intel_syntax)]
#[warn(clippy::inline_asm_x86_att_syntax)]
mod dont_warn {
    use std::arch::{asm, global_asm};

    pub(super) unsafe fn use_asm() {
        asm!("");
        asm!("", options());
        asm!("", options(nostack));
    }

    global_asm!("");
    global_asm!("", options());
}

fn main() {
    unsafe {
        dont_warn::use_asm();
    }
}
