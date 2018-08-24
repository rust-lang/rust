use super::error::{Error, Result};

pub unsafe fn syscall0(mut a: usize) -> Result<usize> {
    asm!("int 0x80"
        : "={eax}"(a)
        : "{eax}"(a)
        : "memory"
        : "intel", "volatile");

    Error::demux(a)
}

pub unsafe fn syscall1(mut a: usize, b: usize) -> Result<usize> {
    asm!("int 0x80"
        : "={eax}"(a)
        : "{eax}"(a), "{ebx}"(b)
        : "memory"
        : "intel", "volatile");

    Error::demux(a)
}

// Clobbers all registers - special for clone
pub unsafe fn syscall1_clobber(mut a: usize, b: usize) -> Result<usize> {
    asm!("int 0x80"
        : "={eax}"(a)
        : "{eax}"(a), "{ebx}"(b)
        : "memory", "ebx", "ecx", "edx", "esi", "edi"
        : "intel", "volatile");

    Error::demux(a)
}

pub unsafe fn syscall2(mut a: usize, b: usize, c: usize) -> Result<usize> {
    asm!("int 0x80"
        : "={eax}"(a)
        : "{eax}"(a), "{ebx}"(b), "{ecx}"(c)
        : "memory"
        : "intel", "volatile");

    Error::demux(a)
}

pub unsafe fn syscall3(mut a: usize, b: usize, c: usize, d: usize) -> Result<usize> {
    asm!("int 0x80"
        : "={eax}"(a)
        : "{eax}"(a), "{ebx}"(b), "{ecx}"(c), "{edx}"(d)
        : "memory"
        : "intel", "volatile");

    Error::demux(a)
}

pub unsafe fn syscall4(mut a: usize, b: usize, c: usize, d: usize, e: usize) -> Result<usize> {
    asm!("int 0x80"
        : "={eax}"(a)
        : "{eax}"(a), "{ebx}"(b), "{ecx}"(c), "{edx}"(d), "{esi}"(e)
        : "memory"
        : "intel", "volatile");

    Error::demux(a)
}

pub unsafe fn syscall5(mut a: usize, b: usize, c: usize, d: usize, e: usize, f: usize)
                       -> Result<usize> {
    asm!("int 0x80"
        : "={eax}"(a)
        : "{eax}"(a), "{ebx}"(b), "{ecx}"(c), "{edx}"(d), "{esi}"(e), "{edi}"(f)
        : "memory"
        : "intel", "volatile");

    Error::demux(a)
}
