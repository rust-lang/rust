// xfail-test not a test

use super::stack::StackSegment;
use core::libc::c_void;
use core::cast::{transmute, transmute_mut_unsafe, transmute_region, transmute_mut_region};

// NOTE: Registers is boxed so that it is 16-byte aligned, for storing SSE regs.
// It would be marginally better not to do this. In C++ we use an attribute on a struct.
pub struct Context(~Registers);

pub impl Context {
    static fn empty() -> Context {
        Context(new_regs())
    }

    /// Create a new context that will resume execution by running ~fn()
    /// # Safety Note
    /// The `start` closure must remain valid for the life of the Task
    static fn new(start: &~fn(), stack: &mut StackSegment) -> Context {

        // The C-ABI function that is the task entry point
        extern fn task_start_wrapper(f: &~fn()) { (*f)() }

        let fp: *c_void = task_start_wrapper as *c_void;
        let argp: *c_void = unsafe { transmute::<&~fn(), *c_void>(&*start) };
        let sp: *uint = stack.end();

        // Save and then immediately load the current context,
        // which we will then modify to call the given function when restored
        let mut regs = new_regs();
        unsafe { swap_registers(transmute_mut_region(&mut *regs), transmute_region(&*regs)) };

        initialize_call_frame(&mut *regs, fp, argp, sp);

        return Context(regs);
    }

    static fn swap(out_context: &mut Context, in_context: &Context) {
        let out_regs: &mut Registers = match out_context { &Context(~ref mut r) => r };
        let in_regs: &Registers = match in_context { &Context(~ref r) => r };

        unsafe { swap_registers(out_regs, in_regs) };
    }
}

extern {
    fn swap_registers(out_regs: *mut Registers, in_regs: *Registers);
}

// Definitions of these registers are in rt/arch/x86_64/regs.h
#[cfg(target_arch = "x86_64")]
type Registers = [uint * 22];

#[cfg(target_arch = "x86_64")]
fn new_regs() -> ~Registers { ~[0, .. 22] }

#[cfg(target_arch = "x86_64")]
fn initialize_call_frame(regs: &mut Registers,
                         fptr: *c_void, arg: *c_void, sp: *uint) {

    // Redefinitions from regs.h
    const RUSTRT_ARG0: uint = 3;
    const RUSTRT_RSP: uint = 1;
    const RUSTRT_IP: uint = 8;
    const RUSTRT_RBP: uint = 2;

    let sp = sp as *uint;
    let sp = align_down(sp);
    let sp = sp.offset(-1);
    let sp = unsafe { transmute_mut_unsafe(sp) };

    // The final return address. 0 indicates the bottom of the stack
    unsafe { *sp = 0; }

    rtdebug!("creating call frame");
    rtdebug!("fptr %x", fptr as uint);
    rtdebug!("arg %x", arg as uint);
    rtdebug!("sp %x", sp as uint);

    regs[RUSTRT_ARG0] = arg as uint;
    regs[RUSTRT_RSP] = sp as uint;
    regs[RUSTRT_IP] = fptr as uint;

    // Last base pointer on the stack should be 0
    regs[RUSTRT_RBP] = 0;
}

fn align_down(sp: *uint) -> *uint {
    unsafe {
        let sp = transmute::<*uint, uint>(sp);
        let sp = sp & !(16 - 1);
        transmute::<uint, *uint>(sp)
    }
}

// NOTE: ptr::offset is positive ints only
#[inline(always)]
pub pure fn offset<T>(ptr: *T, count: int) -> *T {
    use core::sys::size_of;
    unsafe {
        (ptr as int + count * (size_of::<T>() as int)) as *T
    }
}

