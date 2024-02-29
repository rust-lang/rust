//@revisions: i686 x86_64
//@[i686] only-target-i686
//@[x86_64] only-target-x86_64

#[warn(clippy::inline_asm_x86_intel_syntax)]
mod warn_intel {
    use std::arch::{asm, global_asm};

    pub(super) unsafe fn use_asm() {
        asm!("");
        //~^ ERROR: Intel x86 assembly syntax used
        asm!("", options());
        //~^ ERROR: Intel x86 assembly syntax used
        asm!("", options(nostack));
        //~^ ERROR: Intel x86 assembly syntax used
        asm!("", options(att_syntax));
        asm!("", options(nostack, att_syntax));
    }

    global_asm!("");
    //~^ ERROR: Intel x86 assembly syntax used
    global_asm!("", options());
    //~^ ERROR: Intel x86 assembly syntax used
    global_asm!("", options(att_syntax));
}

#[warn(clippy::inline_asm_x86_att_syntax)]
mod warn_att {
    use std::arch::{asm, global_asm};

    pub(super) unsafe fn use_asm() {
        asm!("");
        asm!("", options());
        asm!("", options(nostack));
        asm!("", options(att_syntax));
        //~^ ERROR: AT&T x86 assembly syntax used
        asm!("", options(nostack, att_syntax));
        //~^ ERROR: AT&T x86 assembly syntax used
    }

    global_asm!("");
    global_asm!("", options());
    global_asm!("", options(att_syntax));
    //~^ ERROR: AT&T x86 assembly syntax used
}

fn main() {
    unsafe {
        warn_att::use_asm();
        warn_intel::use_asm();
    }
}
