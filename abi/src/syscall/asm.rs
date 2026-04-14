#[macro_export]
#[cfg(target_arch = "x86_64")]
macro_rules! syscall_asm {
    ($n:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr) => {
        {
            let ret: isize;
            core::arch::asm!(
                "syscall",
                inlateout("rax") $n as usize => ret,
                in("rdi") $a0,
                in("rsi") $a1,
                in("rdx") $a2,
                in("r10") $a3,
                in("r8") $a4,
                in("r9") $a5,
                out("rcx") _,
                out("r11") _,
                options(nostack, preserves_flags)
            );
            ret
        }
    };
}

#[macro_export]
#[cfg(target_arch = "aarch64")]
macro_rules! syscall_asm {
    ($n:expr, $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr) => {
        {
            let ret: isize;
            core::arch::asm!(
                "svc #0",
                inlateout("x8") $n as usize => ret,
                in("x0") $a0,
                in("x1") $a1,
                in("x2") $a2,
                in("x3") $a3,
                in("x4") $a4,
                in("x5") $a5,
                options(nostack, preserves_flags)
            );
            ret
        }
    };
}
