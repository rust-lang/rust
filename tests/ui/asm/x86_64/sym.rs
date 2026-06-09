//@ only-x86_64
//@ only-linux
//@ needs-asm-support
//@ run-pass

#![feature(thread_local)]

use std::arch::asm;

extern "C" fn f1() -> i32 {
    111
}

// The compiler will generate a shim to hide the caller location parameter.
#[track_caller]
fn f2() -> i32 {
    222
}

macro_rules! call {
    ($func:path) => {
        unsafe {
            let result: i32;
            asm!("call {}", sym $func,
                out("rax") result,
                out("rcx") _, out("rdx") _, out("rdi") _, out("rsi") _,
                out("r8") _, out("r9") _, out("r10") _, out("r11") _,
                out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
                out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
                out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
                out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
            );
            result
        }
    }
}

macro_rules! static_addr {
    ($s:expr) => {
        unsafe {
            let result: *const u32;
            // LEA performs a RIP-relative address calculation and returns the address
            asm!("lea {}, [rip + {}]", out(reg) result, sym $s);
            result
        }
    }
}
macro_rules! static_tls_addr {
    ($s:expr) => {
        unsafe {
            let result: *const u32;
            asm!(
                "
                    # Load TLS base address
                    mov {out}, qword ptr fs:[0]
                    # Calculate the address of sym in the TLS block. The @tpoff
                    # relocation gives the offset of the symbol from the start
                    # of the TLS block.
                    lea {out}, [{out} + {sym}@tpoff]
                ",
                out = out(reg) result,
                sym = sym $s
            );
            result
        }
    }
}

static S1: u32 = 111;
#[thread_local]
static S2: u32 = 222;

fn main() {
    assert_eq!(call!(f1), 111);
    assert_eq!(call!(f2), 222);
    assert_eq!(static_addr!(S1), &S1 as *const u32);
    assert_eq!(static_tls_addr!(S2), &S2 as *const u32);
    std::thread::spawn(|| {
        assert_eq!(static_addr!(S1), &S1 as *const u32);
        assert_eq!(static_tls_addr!(S2), &S2 as *const u32);
    })
    .join()
    .unwrap();
}
