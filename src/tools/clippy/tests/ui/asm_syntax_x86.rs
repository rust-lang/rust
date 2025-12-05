//@only-target: i686 x86_64

#[warn(clippy::inline_asm_x86_intel_syntax)]
mod warn_intel {
    use std::arch::{asm, global_asm};

    pub(super) unsafe fn use_asm() {
        unsafe {
            asm!("");
            //~^ inline_asm_x86_intel_syntax

            asm!("", options());
            //~^ inline_asm_x86_intel_syntax

            asm!("", options(nostack));
            //~^ inline_asm_x86_intel_syntax

            asm!("", options(att_syntax));
            asm!("", options(nostack, att_syntax));
        }
    }

    global_asm!("");
    //~^ inline_asm_x86_intel_syntax

    global_asm!("", options());
    //~^ inline_asm_x86_intel_syntax

    global_asm!("", options(att_syntax));
}

#[warn(clippy::inline_asm_x86_att_syntax)]
mod warn_att {
    use std::arch::{asm, global_asm};

    pub(super) unsafe fn use_asm() {
        unsafe {
            asm!("");
            asm!("", options());
            asm!("", options(nostack));
            asm!("", options(att_syntax));
            //~^ inline_asm_x86_att_syntax

            asm!("", options(nostack, att_syntax));
            //~^ inline_asm_x86_att_syntax
        }
    }

    global_asm!("");
    global_asm!("", options());
    global_asm!("", options(att_syntax));
    //~^ inline_asm_x86_att_syntax
}

fn main() {
    unsafe {
        warn_att::use_asm();
        warn_intel::use_asm();
    }
}
