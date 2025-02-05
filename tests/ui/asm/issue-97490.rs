//@ check-pass
//@ only-x86_64
//@ needs-asm-support

#[allow(improper_ctypes_definitions)]  // it's worried about invalid pointers being given as the
                                       // argument value
pub type Yes = extern "sysv64" fn(&'static u8) -> !;

fn main() {
    unsafe {
        let yes = &6 as *const _ as *const Yes;
        core::arch::asm!("call {}", in(reg) yes, options(noreturn));
    }
}
