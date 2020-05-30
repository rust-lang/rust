// no-system-llvm
// only-x86_64
// run-pass

#![feature(asm, track_caller)]

extern "C" fn f1() -> i32 {
    111
}

// The compiler will generate a shim to hide the caller location parameter.
#[track_caller]
fn f2() -> i32 {
    222
}

macro_rules! call {
    ($func:path) => {{
        let result: i32;
        unsafe {
            asm!("call {}", sym $func,
                out("rax") result,
                out("rcx") _, out("rdx") _, out("rdi") _, out("rsi") _,
                out("r8") _, out("r9") _, out("r10") _, out("r11") _,
                out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
                out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
                out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
                out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
            );
        }
        result
    }}
}

fn main() {
    assert_eq!(call!(f1), 111);
    assert_eq!(call!(f2), 222);
}
