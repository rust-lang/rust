// only-aarch64
// only-linux
// run-pass

#![feature(asm, thread_local, asm_sym)]

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
            asm!("bl {}", sym $func,
                out("w0") result,
                out("x20") _, out("x21") _, out("x22") _,
                out("x23") _, out("x24") _, out("x25") _,
                out("x26") _, out("x27") _, out("x28") _,
            );
            result
        }
    }
}

macro_rules! static_addr {
    ($s:expr) => {
        unsafe {
            let result: *const u32;
            asm!(
                // ADRP gives the address of a 4KB page from a PC-relative address
                "adrp {out}, {sym}",
                // We then add the remaining lower 12 bits
                "add {out}, {out}, #:lo12:{sym}",
                out = out(reg) result,
                sym = sym $s);
            result
        }
    }
}
macro_rules! static_tls_addr {
    ($s:expr) => {
        unsafe {
            let result: *const u32;
            asm!(
                // Load the thread pointer register
                "mrs {out}, TPIDR_EL0",
                // Add the top 12 bits of the symbol's offset
                "add {out}, {out}, :tprel_hi12:{sym}",
                // And the bottom 12 bits
                "add {out}, {out}, :tprel_lo12_nc:{sym}",
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
