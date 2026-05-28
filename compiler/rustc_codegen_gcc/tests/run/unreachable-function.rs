// Compiler:
//
// Run-time:
//   status: 0

use std::arch::asm;

fn exit_syscall(status: i32) -> ! {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        asm!(
            "syscall",
            in("rax") 60,
            in("rdi") status,
            options(noreturn)
        );
    }

    #[cfg(not(target_arch = "x86_64"))]
    std::process::exit(status);
}

fn main() {
    // Used to crash with rustc_codegen_gcc.
    exit_syscall(0);
    std::process::exit(1);
}
