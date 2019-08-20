
#[inline(always)]
pub unsafe fn syscall0(n: usize) -> usize {
    let ret : usize;
    asm!("svc 0"   : "={x0}"(ret)
                   : "{x8}"(n)
                   : "memory" "cc"
                   : "volatile");
    ret
}

#[inline(always)]
pub unsafe fn syscall1(n: usize, a1: usize) -> usize {
    let ret : usize;
    asm!("svc 0"   : "={x0}"(ret)
                   : "{x8}"(n), "{x0}"(a1)
                   : "memory" "cc"
                   : "volatile");
    ret
}

#[inline(always)]
pub unsafe fn syscall2(n: usize, a1: usize, a2: usize) -> usize {
    let ret : usize;
    asm!("svc 0"   : "={x0}"(ret)
                   : "{x8}"(n), "{x0}"(a1), "{x1}"(a2)
                   : "memory" "cc"
                   : "volatile");
    ret
}

#[inline(always)]
pub unsafe fn syscall3(n: usize, a1: usize, a2: usize, a3: usize) -> usize {
    let ret : usize;
    asm!("svc 0"   : "={x0}"(ret)
                   : "{x8}"(n), "{x0}"(a1), "{x1}"(a2), "{x2}"(a3)
                   : "memory" "cc"
                   : "volatile");
    ret
}

#[inline(always)]
pub unsafe fn syscall4(n: usize, a1: usize, a2: usize, a3: usize,
                                a4: usize) -> usize {
    let ret : usize;
    asm!("svc 0"   : "={x0}"(ret)
                   : "{x8}"(n), "{x0}"(a1), "{x1}"(a2), "{x2}"(a3), "{x3}"(a4)
                   : "memory" "cc"
                   : "volatile");
    ret
}

#[inline(always)]
pub unsafe fn syscall5(n: usize, a1: usize, a2: usize, a3: usize,
                                a4: usize, a5: usize) -> usize {
    let ret : usize;
    asm!("svc 0"   : "={x0}"(ret)
                   : "{x8}"(n), "{x0}"(a1), "{x1}"(a2), "{x2}"(a3), "{x3}"(a4),
                     "{x4}"(a5)
                   : "memory" "cc"
                   : "volatile");
    ret
}

#[inline(always)]
pub unsafe fn syscall6(n: usize, a1: usize, a2: usize, a3: usize,
                                a4: usize, a5: usize, a6: usize) -> usize {
    let ret : usize;
    asm!("svc 0"   : "={x0}"(ret)
                   : "{x8}"(n), "{x0}"(a1), "{x1}"(a2), "{x2}"(a3), "{x3}"(a4),
                     "{x4}"(a5), "{x6}"(a6)
                   : "memory" "cc"
                   : "volatile");
    ret
}

#[inline(always)]
pub unsafe fn syscall7(n: usize, a1: usize, a2: usize, a3: usize,
                                a4: usize, a5: usize, a6: usize) -> usize {
    let ret : usize;
    asm!("svc 0"   : "={x0}"(ret)
                   : "{x8}"(n), "{x0}"(a1), "{x1}"(a2), "{x2}"(a3), "{x3}"(a4)
                     "{x4}"(a5), "{x6}"(a6)
                   : "memory" "cc"
                   : "volatile");
    ret
}
