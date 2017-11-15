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
#[cfg(all(target_arch = "x86_64", not(feature = "mangled-names")))]
pub unsafe extern fn __rust_probestack() {
    // Our goal here is to touch each page between %rsp+8 and %rsp+8-%rax,
    // ensuring that if any pages are unmapped we'll make a page fault.
    //
    // The ABI here is that the stack frame size is located in `%eax`. Upon
    // return we're not supposed to modify `%esp` or `%eax`.
    asm!("
        mov    %rax,%r11        // duplicate %rax as we're clobbering %r11

        // Main loop, taken in one page increments. We're decrementing rsp by
        // a page each time until there's less than a page remaining. We're
        // guaranteed that this function isn't called unless there's more than a
        // page needed.
        //
        // Note that we're also testing against `8(%rsp)` to account for the 8
        // bytes pushed on the stack orginally with our return address. Using
        // `8(%rsp)` simulates us testing the stack pointer in the caller's
        // context.
    2:
        sub    $$0x1000,%rsp
        test   %rsp,8(%rsp)
        sub    $$0x1000,%r11
        cmp    $$0x1000,%r11
        ja     2b

        // Finish up the last remaining stack space requested, getting the last
        // bits out of r11
        sub    %r11,%rsp
        test   %rsp,8(%rsp)

        // Restore the stack pointer to what it previously was when entering
        // this function. The caller will readjust the stack pointer after we
        // return.
        add    %rax,%rsp

        ret
    " ::: "memory" : "volatile");
    ::core::intrinsics::unreachable();
}

#[naked]
#[no_mangle]
#[cfg(all(target_arch = "x86", not(feature = "mangled-names")))]
pub unsafe extern fn __rust_probestack() {
    // This is the same as x86_64 above, only translated for 32-bit sizes. Note
    // that on Unix we're expected to restore everything as it was, this
    // function basically can't tamper with anything.
    //
    // The ABI here is the same as x86_64, except everything is 32-bits large.
    asm!("
        push   %ecx
        mov    %eax,%ecx
    2:
        sub    $$0x1000,%esp
        test   %esp,8(%esp)
        sub    $$0x1000,%ecx
        cmp    $$0x1000,%ecx
        ja     2b

        sub    %ecx,%esp
        test   %esp,8(%esp)

        add    %eax,%esp
        pop    %ecx
        ret
    " ::: "memory" : "volatile");
    ::core::intrinsics::unreachable();
}
