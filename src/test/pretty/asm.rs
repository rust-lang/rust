#![feature(asm)]

// pretty-mode:expanded
// pp-exact:asm.pp
// only-x86_64

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
        asm!("", out("al") _, lateout("rbx") _);
    }
}
