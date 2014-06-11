// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use stack::Stack;
use std::uint;
use std::mem::transmute;
use std::rt::stack;
use std::raw;

// FIXME #7761: Registers is boxed so that it is 16-byte aligned, for storing
// SSE regs.  It would be marginally better not to do this. In C++ we
// use an attribute on a struct.
// FIXME #7761: It would be nice to define regs as `Box<Option<Registers>>`
// since the registers are sometimes empty, but the discriminant would
// then misalign the regs again.
pub struct Context {
    /// Hold the registers while the task or scheduler is suspended
    regs: Box<Registers>,
    /// Lower bound and upper bound for the stack
    stack_bounds: Option<(uint, uint)>,
}

pub type InitFn = extern "C" fn(uint, *(), *()) -> !;

impl Context {
    pub fn empty() -> Context {
        Context {
            regs: new_regs(),
            stack_bounds: None,
        }
    }

    /// Create a new context that will resume execution by running proc()
    ///
    /// The `init` function will be run with `arg` and the `start` procedure
    /// split up into code and env pointers. It is required that the `init`
    /// function never return.
    ///
    /// FIXME: this is basically an awful the interface. The main reason for
    ///        this is to reduce the number of allocations made when a green
    ///        task is spawned as much as possible
    pub fn new(init: InitFn, arg: uint, start: proc():Send,
               stack: &mut Stack) -> Context {

        let sp: *uint = stack.end();
        let sp: *mut uint = sp as *mut uint;
        // Save and then immediately load the current context,
        // which we will then modify to call the given function when restored
        let mut regs = new_regs();

        initialize_call_frame(&mut *regs,
                              init,
                              arg,
                              unsafe { transmute(start) },
                              sp);

        // Scheduler tasks don't have a stack in the "we allocated it" sense,
        // but rather they run on pthreads stacks. We have complete control over
        // them in terms of the code running on them (and hopefully they don't
        // overflow). Additionally, their coroutine stacks are listed as being
        // zero-length, so that's how we detect what's what here.
        let stack_base: *uint = stack.start();
        let bounds = if sp as uint == stack_base as uint {
            None
        } else {
            Some((stack_base as uint, sp as uint))
        };
        return Context {
            regs: regs,
            stack_bounds: bounds,
        }
    }

    /* Switch contexts

    Suspend the current execution context and resume another by
    saving the registers values of the executing thread to a Context
    then loading the registers from a previously saved Context.
    */
    pub fn swap(out_context: &mut Context, in_context: &Context) {
        rtdebug!("swapping contexts");
        let out_regs: &mut Registers = match out_context {
            &Context { regs: box ref mut r, .. } => r
        };
        let in_regs: &Registers = match in_context {
            &Context { regs: box ref r, .. } => r
        };

        rtdebug!("noting the stack limit and doing raw swap");

        unsafe {
            // Right before we switch to the new context, set the new context's
            // stack limit in the OS-specified TLS slot. This also  means that
            // we cannot call any more rust functions after record_stack_bounds
            // returns because they would all likely fail due to the limit being
            // invalid for the current task. Lucky for us `rust_swap_registers`
            // is a C function so we don't have to worry about that!
            match in_context.stack_bounds {
                Some((lo, hi)) => stack::record_stack_bounds(lo, hi),
                // If we're going back to one of the original contexts or
                // something that's possibly not a "normal task", then reset
                // the stack limit to 0 to make morestack never fail
                None => stack::record_stack_bounds(0, uint::MAX),
            }
            rust_swap_registers(out_regs, in_regs)
        }
    }
}

#[link(name = "context_switch", kind = "static")]
extern {
    fn rust_swap_registers(out_regs: *mut Registers, in_regs: *Registers);
}

// Register contexts used in various architectures
//
// These structures all represent a context of one task throughout its
// execution. Each struct is a representation of the architecture's register
// set. When swapping between tasks, these register sets are used to save off
// the current registers into one struct, and load them all from another.
//
// Note that this is only used for context switching, which means that some of
// the registers may go unused. For example, for architectures with
// callee/caller saved registers, the context will only reflect the callee-saved
// registers. This is because the caller saved registers are already stored
// elsewhere on the stack (if it was necessary anyway).
//
// Additionally, there may be fields on various architectures which are unused
// entirely because they only reflect what is theoretically possible for a
// "complete register set" to show, but user-space cannot alter these registers.
// An example of this would be the segment selectors for x86.
//
// These structures/functions are roughly in-sync with the source files inside
// of src/rt/arch/$arch. The only currently used function from those folders is
// the `rust_swap_registers` function, but that's only because for now segmented
// stacks are disabled.

#[cfg(target_arch = "x86")]
#[repr(C)]
struct Registers {
    eax: u32, ebx: u32, ecx: u32, edx: u32,
    ebp: u32, esi: u32, edi: u32, esp: u32,
    cs: u16, ds: u16, ss: u16, es: u16, fs: u16, gs: u16,
    eflags: u32, eip: u32
}

#[cfg(target_arch = "x86")]
fn new_regs() -> Box<Registers> {
    box Registers {
        eax: 0, ebx: 0, ecx: 0, edx: 0,
        ebp: 0, esi: 0, edi: 0, esp: 0,
        cs: 0, ds: 0, ss: 0, es: 0, fs: 0, gs: 0,
        eflags: 0, eip: 0
    }
}

#[cfg(target_arch = "x86")]
fn initialize_call_frame(regs: &mut Registers, fptr: InitFn, arg: uint,
                         procedure: raw::Procedure, sp: *mut uint) {

    // x86 has interesting stack alignment requirements, so do some alignment
    // plus some offsetting to figure out what the actual stack should be.
    let sp = align_down(sp);
    let sp = mut_offset(sp, -4);

    unsafe { *mut_offset(sp, 2) = procedure.env as uint };
    unsafe { *mut_offset(sp, 1) = procedure.code as uint };
    unsafe { *mut_offset(sp, 0) = arg as uint };
    let sp = mut_offset(sp, -1);
    unsafe { *sp = 0 }; // The final return address

    regs.esp = sp as u32;
    regs.eip = fptr as u32;

    // Last base pointer on the stack is 0
    regs.ebp = 0;
}

// windows requires saving more registers (both general and XMM), so the windows
// register context must be larger.
#[cfg(windows, target_arch = "x86_64")]
type Registers = [uint, ..34];
#[cfg(not(windows), target_arch = "x86_64")]
type Registers = [uint, ..22];

#[cfg(windows, target_arch = "x86_64")]
fn new_regs() -> Box<Registers> { box() ([0, .. 34]) }
#[cfg(not(windows), target_arch = "x86_64")]
fn new_regs() -> Box<Registers> { box() ([0, .. 22]) }

#[cfg(target_arch = "x86_64")]
fn initialize_call_frame(regs: &mut Registers, fptr: InitFn, arg: uint,
                         procedure: raw::Procedure, sp: *mut uint) {
    extern { fn rust_bootstrap_green_task(); }

    // Redefinitions from rt/arch/x86_64/regs.h
    static RUSTRT_RSP: uint = 1;
    static RUSTRT_IP: uint = 8;
    static RUSTRT_RBP: uint = 2;
    static RUSTRT_R12: uint = 4;
    static RUSTRT_R13: uint = 5;
    static RUSTRT_R14: uint = 6;
    static RUSTRT_R15: uint = 7;

    let sp = align_down(sp);
    let sp = mut_offset(sp, -1);

    // The final return address. 0 indicates the bottom of the stack
    unsafe { *sp = 0; }

    rtdebug!("creating call frame");
    rtdebug!("fptr {:#x}", fptr as uint);
    rtdebug!("arg {:#x}", arg);
    rtdebug!("sp {}", sp);

    // These registers are frobbed by rust_bootstrap_green_task into the right
    // location so we can invoke the "real init function", `fptr`.
    regs[RUSTRT_R12] = arg as uint;
    regs[RUSTRT_R13] = procedure.code as uint;
    regs[RUSTRT_R14] = procedure.env as uint;
    regs[RUSTRT_R15] = fptr as uint;

    // These registers are picked up by the regular context switch paths. These
    // will put us in "mostly the right context" except for frobbing all the
    // arguments to the right place. We have the small trampoline code inside of
    // rust_bootstrap_green_task to do that.
    regs[RUSTRT_RSP] = sp as uint;
    regs[RUSTRT_IP] = rust_bootstrap_green_task as uint;

    // Last base pointer on the stack should be 0
    regs[RUSTRT_RBP] = 0;
}

#[cfg(target_arch = "arm")]
type Registers = [uint, ..32];

#[cfg(target_arch = "arm")]
fn new_regs() -> Box<Registers> { box {[0, .. 32]} }

#[cfg(target_arch = "arm")]
fn initialize_call_frame(regs: &mut Registers, fptr: InitFn, arg: uint,
                         procedure: raw::Procedure, sp: *mut uint) {
    extern { fn rust_bootstrap_green_task(); }

    let sp = align_down(sp);
    // sp of arm eabi is 8-byte aligned
    let sp = mut_offset(sp, -2);

    // The final return address. 0 indicates the bottom of the stack
    unsafe { *sp = 0; }

    // ARM uses the same technique as x86_64 to have a landing pad for the start
    // of all new green tasks. Neither r1/r2 are saved on a context switch, so
    // the shim will copy r3/r4 into r1/r2 and then execute the function in r5
    regs[0] = arg as uint;              // r0
    regs[3] = procedure.code as uint;   // r3
    regs[4] = procedure.env as uint;    // r4
    regs[5] = fptr as uint;             // r5
    regs[13] = sp as uint;                          // #52 sp, r13
    regs[14] = rust_bootstrap_green_task as uint;   // #56 pc, r14 --> lr
}

#[cfg(target_arch = "mips")]
type Registers = [uint, ..32];

#[cfg(target_arch = "mips")]
fn new_regs() -> Box<Registers> { box {[0, .. 32]} }

#[cfg(target_arch = "mips")]
fn initialize_call_frame(regs: &mut Registers, fptr: InitFn, arg: uint,
                         procedure: raw::Procedure, sp: *mut uint) {
    let sp = align_down(sp);
    // sp of mips o32 is 8-byte aligned
    let sp = mut_offset(sp, -2);

    // The final return address. 0 indicates the bottom of the stack
    unsafe { *sp = 0; }

    regs[4] = arg as uint;
    regs[5] = procedure.code as uint;
    regs[6] = procedure.env as uint;
    regs[29] = sp as uint;
    regs[25] = fptr as uint;
    regs[31] = fptr as uint;
}

fn align_down(sp: *mut uint) -> *mut uint {
    let sp = (sp as uint) & !(16 - 1);
    sp as *mut uint
}

// ptr::mut_offset is positive ints only
#[inline]
pub fn mut_offset<T>(ptr: *mut T, count: int) -> *mut T {
    use std::mem::size_of;
    (ptr as int + count * (size_of::<T>() as int)) as *mut T
}
