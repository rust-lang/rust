// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module defines the `__rust_probestack` intrinsic which is used in the
//! implementation of "stack probes" on certain platforms.
//!
//! The purpose of a stack probe is to provide a static guarantee that if a
//! thread has a guard page then a stack overflow is guaranteed to hit that
//! guard page. If a function did not have a stack probe then there's a risk of
//! having a stack frame *larger* than the guard page, so a function call could
//! skip over the guard page entirely and then later hit maybe the heap or
//! another thread, possibly leading to security vulnerabilities such as [The
//! Stack Clash], for example.
//!
//! [The Stack Clash]: https://blog.qualys.com/securitylabs/2017/06/19/the-stack-clash
//!
//! The `__rust_probestack` is called in the prologue of functions whose stack
//! size is larger than the guard page, for example larger than 4096 bytes on
//! x86. This function is then responsible for "touching" all pages relevant to
//! the stack to ensure that that if any of them are the guard page we'll hit
//! them guaranteed.
//!
//! The precise ABI for how this function operates is defined by LLVM. There's
//! no real documentation as to what this is, so you'd basically need to read
//! the LLVM source code for reference. Often though the test cases can be
//! illuminating as to the ABI that's generated, or just looking at the output
//! of `llc`.
//!
//! Note that `#[naked]` is typically used here for the stack probe because the
//! ABI corresponds to no actual ABI.
//!
//! Finally it's worth noting that at the time of this writing LLVM only has
//! support for stack probes on x86 and x86_64. There's no support for stack
//! probes on any other architecture like ARM or PowerPC64. LLVM I'm sure would
//! be more than welcome to accept such a change!

#![cfg(not(windows))] // Windows already has builtins to do this

#[naked]
#[no_mangle]
#[cfg(target_arch = "x86_64")]
pub unsafe extern fn __rust_probestack() {
    // Our goal here is to touch each page between %rsp+8 and %rsp+8-%rax,
    // ensuring that if any pages are unmapped we'll make a page fault.
    //
    // The ABI here is that the stack frame size is located in `%eax`. Upon
    // return we're not supposed to modify `%esp` or `%eax`.
    asm!("
        lea    8(%rsp),%r11     // rsp before calling this routine -> r11

        // Main loop, taken in one page increments. We're decrementing r11 by
        // a page each time until there's less than a page remaining. We're
        // guaranteed that this function isn't called unless there's more than a
        // page needed
    2:
        sub    $$0x1000,%r11
        test   %r11,(%r11)
        sub    $$0x1000,%rax
        cmp    $$0x1000,%rax
        ja     2b

        // Finish up the last remaining stack space requested, getting the last
        // bits out of rax
        sub    %rax,%r11
        test   %r11,(%r11)

        // We now know that %r11 is (%rsp + 8 - %rax) so to recover rax
        // we calculate (%rsp + 8) - %r11 which will give us %rax
        lea    8(%rsp),%rax
        sub    %r11,%rax

        ret
    ");
    ::core::intrinsics::unreachable();
}

#[naked]
#[no_mangle]
#[cfg(target_arch = "x86")]
pub unsafe extern fn __rust_probestack() {
    // This is the same as x86_64 above, only translated for 32-bit sizes. Note
    // that on Unix we're expected to restore everything as it was, this
    // function basically can't tamper with anything.
    //
    // The ABI here is the same as x86_64, except everything is 32-bits large.
    asm!("
        push   %ecx
        lea    8(%esp),%ecx
    2:
        sub    $$0x1000,%ecx
        test   %ecx,(%ecx)
        sub    $$0x1000,%eax
        cmp    $$0x1000,%eax
        ja     2b

        sub    %eax,%ecx
        test   %ecx,(%ecx)

        lea    8(%esp),%eax
        sub    %ecx,%eax
        pop    %ecx
        ret
    ");
    ::core::intrinsics::unreachable();
}
