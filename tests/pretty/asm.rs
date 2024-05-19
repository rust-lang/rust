//@ pretty-mode:expanded
//@ pp-exact:asm.pp
//@ only-x86_64

use std::arch::asm;

pub fn main() {
    let a: i32;
    let mut b = 4i32;
    unsafe {
        asm!("");
        asm!("", options());
        asm!("", options(nostack, nomem));
        asm!("{}", in(reg) 4);
        asm!("{0}", out(reg) a);
        asm!("{name}", name = inout(reg) b);
        asm!("{} {}", out(reg) _, inlateout(reg) b => _);
        asm!("", out("al") _, lateout("rcx") _);
        asm!("inst1", "inst2");
        asm!("inst1 {}, 42", "inst2 {}, 24", in(reg) a, out(reg) b);
        asm!("inst2 {1}, 24", "inst1 {0}, 42", in(reg) a, out(reg) b);
        asm!("inst1 {}, 42", "inst2 {name}, 24", in(reg) a, name = out(reg) b);
        asm!(
            "inst1
inst2"
        );
        asm!("inst1\ninst2");
        asm!("inst1\n\tinst2");
        asm!("inst1\ninst2", "inst3\ninst4");
    }
}
