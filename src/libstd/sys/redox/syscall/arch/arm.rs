use super::error::{Error, Result};

pub unsafe fn syscall0(mut a: usize) -> Result<usize> {
    asm!("swi $$0"
        : "={r0}"(a)
        : "{r7}"(a)
        : "memory"
        : "volatile");

    Error::demux(a)
}

pub unsafe fn syscall1(mut a: usize, b: usize) -> Result<usize> {
    asm!("swi $$0"
        : "={r0}"(a)
        : "{r7}"(a), "{r0}"(b)
        : "memory"
        : "volatile");

    Error::demux(a)
}

// Clobbers all registers - special for clone
pub unsafe fn syscall1_clobber(mut a: usize, b: usize) -> Result<usize> {
    asm!("swi $$0"
        : "={r0}"(a)
        : "{r7}"(a), "{r0}"(b)
        : "memory", "r0", "r1", "r2", "r3", "r4"
        : "volatile");

    Error::demux(a)
}

pub unsafe fn syscall2(mut a: usize, b: usize, c: usize) -> Result<usize> {
    asm!("swi $$0"
        : "={r0}"(a)
        : "{r7}"(a), "{r0}"(b), "{r1}"(c)
        : "memory"
        : "volatile");

    Error::demux(a)
}

pub unsafe fn syscall3(mut a: usize, b: usize, c: usize, d: usize) -> Result<usize> {
    asm!("swi $$0"
        : "={r0}"(a)
        : "{r7}"(a), "{r0}"(b), "{r1}"(c), "{r2}"(d)
        : "memory"
        : "volatile");

    Error::demux(a)
}

pub unsafe fn syscall4(mut a: usize, b: usize, c: usize, d: usize, e: usize) -> Result<usize> {
    asm!("swi $$0"
        : "={r0}"(a)
        : "{r7}"(a), "{r0}"(b), "{r1}"(c), "{r2}"(d), "{r3}"(e)
        : "memory"
        : "volatile");

    Error::demux(a)
}

pub unsafe fn syscall5(mut a: usize, b: usize, c: usize, d: usize, e: usize, f: usize)
                       -> Result<usize> {
    asm!("swi $$0"
        : "={r0}"(a)
        : "{r7}"(a), "{r0}"(b), "{r1}"(c), "{r2}"(d), "{r3}"(e), "{r4}"(f)
        : "memory"
        : "volatile");

    Error::demux(a)
}
