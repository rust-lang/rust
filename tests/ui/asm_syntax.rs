#![feature(asm)]
// only-x86_64

#[warn(clippy::inline_asm_x86_intel_syntax)]
mod warn_intel {
    pub(super) unsafe fn use_asm() {
        asm!("");
        asm!("", options());
        asm!("", options(nostack));
        asm!("", options(att_syntax));
        asm!("", options(nostack, att_syntax));
    }
}

#[warn(clippy::inline_asm_x86_att_syntax)]
mod warn_att {
    pub(super) unsafe fn use_asm() {
        asm!("");
        asm!("", options());
        asm!("", options(nostack));
        asm!("", options(att_syntax));
        asm!("", options(nostack, att_syntax));
    }
}

fn main() {
    unsafe {
        warn_att::use_asm();
        warn_intel::use_asm();
    }
}
