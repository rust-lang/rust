// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::error::{Error, Result};

pub unsafe fn syscall0(mut a: usize) -> Result<usize> {
    asm!("int 0x80"
        : "={rax}"(a)
        : "{rax}"(a)
        : "memory"
        : "intel", "volatile");

    Error::demux(a)
}

pub unsafe fn syscall1(mut a: usize, b: usize) -> Result<usize> {
    asm!("int 0x80"
        : "={rax}"(a)
        : "{rax}"(a), "{rbx}"(b)
        : "memory"
        : "intel", "volatile");

    Error::demux(a)
}

// Clobbers all registers - special for clone
pub unsafe fn syscall1_clobber(mut a: usize, b: usize) -> Result<usize> {
    asm!("int 0x80"
        : "={rax}"(a)
        : "{rax}"(a), "{rbx}"(b)
        : "memory", "rbx", "rcx", "rdx", "rsi", "rdi", "r8",
          "r9", "r10", "r11", "r12", "r13", "r14", "r15"
        : "intel", "volatile");

    Error::demux(a)
}

pub unsafe fn syscall2(mut a: usize, b: usize, c: usize) -> Result<usize> {
    asm!("int 0x80"
        : "={rax}"(a)
        : "{rax}"(a), "{rbx}"(b), "{rcx}"(c)
        : "memory"
        : "intel", "volatile");

    Error::demux(a)
}

pub unsafe fn syscall3(mut a: usize, b: usize, c: usize, d: usize) -> Result<usize> {
    asm!("int 0x80"
        : "={rax}"(a)
        : "{rax}"(a), "{rbx}"(b), "{rcx}"(c), "{rdx}"(d)
        : "memory"
        : "intel", "volatile");

    Error::demux(a)
}

pub unsafe fn syscall4(mut a: usize, b: usize, c: usize, d: usize, e: usize) -> Result<usize> {
    asm!("int 0x80"
        : "={rax}"(a)
        : "{rax}"(a), "{rbx}"(b), "{rcx}"(c), "{rdx}"(d), "{rsi}"(e)
        : "memory"
        : "intel", "volatile");

    Error::demux(a)
}

pub unsafe fn syscall5(mut a: usize, b: usize, c: usize, d: usize, e: usize, f: usize)
                       -> Result<usize> {
    asm!("int 0x80"
        : "={rax}"(a)
        : "{rax}"(a), "{rbx}"(b), "{rcx}"(c), "{rdx}"(d), "{rsi}"(e), "{rdi}"(f)
        : "memory"
        : "intel", "volatile");

    Error::demux(a)
}
