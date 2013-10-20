// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::*;
use super::stack::StackSegment;
use libc::c_void;
use uint;
use cast::{transmute, transmute_mut_unsafe,
           transmute_region, transmute_mut_region};

pub static RED_ZONE: uint = 20 * 1024;

// FIXME #7761: Registers is boxed so that it is 16-byte aligned, for storing
// SSE regs.  It would be marginally better not to do this. In C++ we
// use an attribute on a struct.
// FIXME #7761: It would be nice to define regs as `~Option<Registers>` since
// the registers are sometimes empty, but the discriminant would
// then misalign the regs again.
pub struct Context {
    /// The context entry point, saved here for later destruction
    priv start: Option<~~fn()>,
    /// Hold the registers while the task or scheduler is suspended
    priv regs: ~Registers,
    /// Lower bound and upper bound for the stack
    priv stack_bounds: Option<(uint, uint)>,
}

impl Context {
    pub fn empty() -> Context {
        Context {
            start: None,
            regs: new_regs(),
            stack_bounds: None,
        }
    }

    /// Create a new context that will resume execution by running ~fn()
    pub fn new(start: ~fn(), stack: &mut StackSegment) -> Context {
        // FIXME #7767: Putting main into a ~ so it's a thin pointer and can
        // be passed to the spawn function.  Another unfortunate
        // allocation
        let start = ~start;

        // The C-ABI function that is the task entry point
        extern fn task_start_wrapper(f: &~fn()) { (*f)() }

        let fp: *c_void = task_start_wrapper as *c_void;
        let argp: *c_void = unsafe { transmute::<&~fn(), *c_void>(&*start) };
        let sp: *uint = stack.end();
        let sp: *mut uint = unsafe { transmute_mut_unsafe(sp) };
        // Save and then immediately load the current context,
        // which we will then modify to call the given function when restored
        let mut regs = new_regs();
        unsafe {
            swap_registers(transmute_mut_region(&mut *regs), transmute_region(&*regs));
        };

        initialize_call_frame(&mut *regs, fp, argp, sp);

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
            start: Some(start),
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
            &Context { regs: ~ref mut r, _ } => r
        };
        let in_regs: &Registers = match in_context {
            &Context { regs: ~ref r, _ } => r
        };

        rtdebug!("noting the stack limit and doing raw swap");

        unsafe {
            // Right before we switch to the new context, set the new context's
            // stack limit in the OS-specified TLS slot. This also  means that
            // we cannot call any more rust functions after record_stack_bounds
            // returns because they would all likely fail due to the limit being
            // invalid for the current task. Lucky for us `swap_registers` is a
            // C function so we don't have to worry about that!
            match in_context.stack_bounds {
                Some((lo, hi)) => record_stack_bounds(lo, hi),
                // If we're going back to one of the original contexts or
                // something that's possibly not a "normal task", then reset
                // the stack limit to 0 to make morestack never fail
                None => record_stack_bounds(0, uint::max_value),
            }
            swap_registers(out_regs, in_regs)
        }
    }
}

extern {
    #[rust_stack]
    fn swap_registers(out_regs: *mut Registers, in_regs: *Registers);
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
// the `swap_registers` function, but that's only because for now segmented
// stacks are disabled.

#[cfg(target_arch = "x86")]
struct Registers {
    eax: u32, ebx: u32, ecx: u32, edx: u32,
    ebp: u32, esi: u32, edi: u32, esp: u32,
    cs: u16, ds: u16, ss: u16, es: u16, fs: u16, gs: u16,
    eflags: u32, eip: u32
}

#[cfg(target_arch = "x86")]
fn new_regs() -> ~Registers {
    ~Registers {
        eax: 0, ebx: 0, ecx: 0, edx: 0,
        ebp: 0, esi: 0, edi: 0, esp: 0,
        cs: 0, ds: 0, ss: 0, es: 0, fs: 0, gs: 0,
        eflags: 0, eip: 0
    }
}

#[cfg(target_arch = "x86")]
fn initialize_call_frame(regs: &mut Registers, fptr: *c_void, arg: *c_void,
                         sp: *mut uint) {

    let sp = align_down(sp);
    let sp = mut_offset(sp, -4);

    unsafe { *sp = arg as uint };
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
fn new_regs() -> ~Registers { ~([0, .. 34]) }
#[cfg(not(windows), target_arch = "x86_64")]
fn new_regs() -> ~Registers { ~([0, .. 22]) }

#[cfg(target_arch = "x86_64")]
fn initialize_call_frame(regs: &mut Registers, fptr: *c_void, arg: *c_void,
                         sp: *mut uint) {

    // Redefinitions from rt/arch/x86_64/regs.h
    static RUSTRT_ARG0: uint = 3;
    static RUSTRT_RSP: uint = 1;
    static RUSTRT_IP: uint = 8;
    static RUSTRT_RBP: uint = 2;

    let sp = align_down(sp);
    let sp = mut_offset(sp, -1);

    // The final return address. 0 indicates the bottom of the stack
    unsafe { *sp = 0; }

    rtdebug!("creating call frame");
    rtdebug!("fptr {}", fptr);
    rtdebug!("arg {}", arg);
    rtdebug!("sp {}", sp);

    regs[RUSTRT_ARG0] = arg as uint;
    regs[RUSTRT_RSP] = sp as uint;
    regs[RUSTRT_IP] = fptr as uint;

    // Last base pointer on the stack should be 0
    regs[RUSTRT_RBP] = 0;
}

#[cfg(target_arch = "arm")]
type Registers = [uint, ..32];

#[cfg(target_arch = "arm")]
fn new_regs() -> ~Registers { ~([0, .. 32]) }

#[cfg(target_arch = "arm")]
fn initialize_call_frame(regs: &mut Registers, fptr: *c_void, arg: *c_void,
                         sp: *mut uint) {
    let sp = align_down(sp);
    // sp of arm eabi is 8-byte aligned
    let sp = mut_offset(sp, -2);

    // The final return address. 0 indicates the bottom of the stack
    unsafe { *sp = 0; }

    regs[0] = arg as uint;   // r0
    regs[13] = sp as uint;   // #53 sp, r13
    regs[14] = fptr as uint; // #60 pc, r15 --> lr
}

#[cfg(target_arch = "mips")]
type Registers = [uint, ..32];

#[cfg(target_arch = "mips")]
fn new_regs() -> ~Registers { ~([0, .. 32]) }

#[cfg(target_arch = "mips")]
fn initialize_call_frame(regs: &mut Registers, fptr: *c_void, arg: *c_void,
                         sp: *mut uint) {
    let sp = align_down(sp);
    // sp of mips o32 is 8-byte aligned
    let sp = mut_offset(sp, -2);

    // The final return address. 0 indicates the bottom of the stack
    unsafe { *sp = 0; }

    regs[4] = arg as uint;
    regs[29] = sp as uint;
    regs[25] = fptr as uint;
    regs[31] = fptr as uint;
}

fn align_down(sp: *mut uint) -> *mut uint {
    unsafe {
        let sp: uint = transmute(sp);
        let sp = sp & !(16 - 1);
        transmute::<uint, *mut uint>(sp)
    }
}

// ptr::mut_offset is positive ints only
#[inline]
pub fn mut_offset<T>(ptr: *mut T, count: int) -> *mut T {
    use mem::size_of;
    (ptr as int + count * (size_of::<T>() as int)) as *mut T
}

#[inline(always)]
pub unsafe fn record_stack_bounds(stack_lo: uint, stack_hi: uint) {
    // When the old runtime had segmented stacks, it used a calculation that was
    // "limit + RED_ZONE + FUDGE". The red zone was for things like dynamic
    // symbol resolution, llvm function calls, etc. In theory this red zone
    // value is 0, but it matters far less when we have gigantic stacks because
    // we don't need to be so exact about our stack budget. The "fudge factor"
    // was because LLVM doesn't emit a stack check for functions < 256 bytes in
    // size. Again though, we have giant stacks, so we round all these
    // calculations up to the nice round number of 20k.
    record_sp_limit(stack_lo + RED_ZONE);

    return target_record_stack_bounds(stack_lo, stack_hi);

    #[cfg(not(windows))] #[cfg(not(target_arch = "x86_64"))] #[inline(always)]
    unsafe fn target_record_stack_bounds(_stack_lo: uint, _stack_hi: uint) {}
    #[cfg(windows, target_arch = "x86_64")] #[inline(always)]
    unsafe fn target_record_stack_bounds(stack_lo: uint, stack_hi: uint) {
        // Windows compiles C functions which may check the stack bounds. This
        // means that if we want to perform valid FFI on windows, then we need
        // to ensure that the stack bounds are what they truly are for this
        // task. More info can be found at:
        //   https://github.com/mozilla/rust/issues/3445#issuecomment-26114839
        //
        // stack range is at TIB: %gs:0x08 (top) and %gs:0x10 (bottom)
        asm!("mov $0, %gs:0x08" :: "r"(stack_lo) :: "volatile");
        asm!("mov $0, %gs:0x10" :: "r"(stack_hi) :: "volatile");
    }
}

/// Records the current limit of the stack as specified by `end`.
///
/// This is stored in an OS-dependent location, likely inside of the thread
/// local storage. The location that the limit is stored is a pre-ordained
/// location because it's where LLVM has emitted code to check.
///
/// Note that this cannot be called under normal circumstances. This function is
/// changing the stack limit, so upon returning any further function calls will
/// possibly be triggering the morestack logic if you're not careful.
///
/// Also note that this and all of the inside functions are all flagged as
/// "inline(always)" because they're messing around with the stack limits.  This
/// would be unfortunate for the functions themselves to trigger a morestack
/// invocation (if they were an actual function call).
#[inline(always)]
pub unsafe fn record_sp_limit(limit: uint) {
    return target_record_sp_limit(limit);

    // x86-64
    #[cfg(target_arch = "x86_64", target_os = "macos")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        asm!("movq $$0x60+90*8, %rsi
              movq $0, %gs:(%rsi)" :: "r"(limit) : "rsi" : "volatile")
    }
    #[cfg(target_arch = "x86_64", target_os = "linux")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        asm!("movq $0, %fs:112" :: "r"(limit) :: "volatile")
    }
    #[cfg(target_arch = "x86_64", target_os = "win32")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        // see: http://en.wikipedia.org/wiki/Win32_Thread_Information_Block
        // store this inside of the "arbitrary data slot", but double the size
        // because this is 64 bit instead of 32 bit
        asm!("movq $0, %gs:0x28" :: "r"(limit) :: "volatile")
    }
    #[cfg(target_arch = "x86_64", target_os = "freebsd")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        asm!("movq $0, %fs:24" :: "r"(limit) :: "volatile")
    }

    // x86
    #[cfg(target_arch = "x86", target_os = "macos")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        asm!("movl $$0x48+90*4, %eax
              movl $0, %gs:(%eax)" :: "r"(limit) : "eax" : "volatile")
    }
    #[cfg(target_arch = "x86", target_os = "linux")]
    #[cfg(target_arch = "x86", target_os = "freebsd")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        asm!("movl $0, %gs:48" :: "r"(limit) :: "volatile")
    }
    #[cfg(target_arch = "x86", target_os = "win32")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        // see: http://en.wikipedia.org/wiki/Win32_Thread_Information_Block
        // store this inside of the "arbitrary data slot"
        asm!("movl $0, %fs:0x14" :: "r"(limit) :: "volatile")
    }

    // mips, arm - Some brave soul can port these to inline asm, but it's over
    //             my head personally
    #[cfg(target_arch = "mips")]
    #[cfg(target_arch = "arm")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        return record_sp_limit(limit as *c_void);
        extern {
            #[rust_stack]
            fn record_sp_limit(limit: *c_void);
        }
    }
}

/// The counterpart of the function above, this function will fetch the current
/// stack limit stored in TLS.
///
/// Note that all of these functions are meant to be exact counterparts of their
/// brethren above, except that the operands are reversed.
///
/// As with the setter, this function does not have a __morestack header and can
/// therefore be called in a "we're out of stack" situation.
#[inline(always)]
// NOTE: after the next snapshot, can remove the initialization before inline
//       assembly due to an improvement in how it's handled, then this specific
//       allow directive should get removed.
#[allow(dead_assignment)]
pub unsafe fn get_sp_limit() -> uint {
    return target_get_sp_limit();

    // x86-64
    #[cfg(target_arch = "x86_64", target_os = "macos")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        let mut limit: uint = 0;
        asm!("movq $$0x60+90*8, %rsi
              movq %gs:(%rsi), $0" : "=r"(limit) :: "rsi" : "volatile");
        return limit;
    }
    #[cfg(target_arch = "x86_64", target_os = "linux")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        let mut limit: uint = 0;
        asm!("movq %fs:112, $0" : "=r"(limit) ::: "volatile");
        return limit;
    }
    #[cfg(target_arch = "x86_64", target_os = "win32")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        let mut limit: uint = 0;
        asm!("movq %gs:0x28, $0" : "=r"(limit) ::: "volatile");
        return limit;
    }
    #[cfg(target_arch = "x86_64", target_os = "freebsd")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        let mut limit: uint = 0;
        asm!("movq %fs:24, $0" : "=r"(limit) ::: "volatile");
        return limit;
    }

    // x86
    #[cfg(target_arch = "x86", target_os = "macos")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        let mut limit: uint = 0;
        asm!("movl $$0x48+90*4, %eax
              movl %gs:(%eax), $0" : "=r"(limit) :: "eax" : "volatile");
        return limit;
    }
    #[cfg(target_arch = "x86", target_os = "linux")]
    #[cfg(target_arch = "x86", target_os = "freebsd")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        let mut limit: uint = 0;
        asm!("movl %gs:48, $0" : "=r"(limit) ::: "volatile");
        return limit;
    }
    #[cfg(target_arch = "x86", target_os = "win32")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        let mut limit: uint = 0;
        asm!("movl %fs:0x14, $0" : "=r"(limit) ::: "volatile");
        return limit;
    }

    // mips, arm - Some brave soul can port these to inline asm, but it's over
    //             my head personally
    #[cfg(target_arch = "mips")]
    #[cfg(target_arch = "arm")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        return get_sp_limit() as uint;
        extern {
            #[rust_stack]
            fn get_sp_limit() -> *c_void;
        }
    }
}
