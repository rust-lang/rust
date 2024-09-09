//@ only-x86_64
//@ compile-flags: -C target-feature=+avx2

use std::arch::asm;

fn main() {
    let mut foo = 0;
    let mut bar = 0;
    unsafe {
        // Bad register/register class

        asm!("{}", in(foo) foo);
        //~^ ERROR invalid register class `foo`: unknown register class
        asm!("", in("foo") foo);
        //~^ ERROR invalid register `foo`: unknown register
        asm!("{:z}", in(reg) foo);
        //~^ ERROR invalid asm template modifier for this register class
        asm!("{:r}", in(xmm_reg) foo);
        //~^ ERROR invalid asm template modifier for this register class
        asm!("{:a}", const 0);
        //~^ ERROR asm template modifiers are not allowed for `const` arguments
        asm!("{:a}", sym main);
        //~^ ERROR asm template modifiers are not allowed for `sym` arguments
        asm!("", in("ebp") foo);
        //~^ ERROR invalid register `ebp`: the frame pointer cannot be used as an operand
        asm!("", in("rsp") foo);
        //~^ ERROR invalid register `rsp`: the stack pointer cannot be used as an operand
        asm!("", in("ip") foo);
        //~^ ERROR invalid register `ip`: the instruction pointer cannot be used as an operand

        asm!("", in("st(2)") foo);
        //~^ ERROR register class `x87_reg` can only be used as a clobber, not as an input or output
        //~| ERROR `i32` cannot be used with this register class
        asm!("", in("mm0") foo);
        //~^ ERROR register class `mmx_reg` can only be used as a clobber, not as an input or output
        //~| ERROR `i32` cannot be used with this register class
        asm!("", in("k0") foo);
        //~^ ERROR register class `kreg0` can only be used as a clobber, not as an input or output
        //~| ERROR `i32` cannot be used with this register class
        asm!("", out("st(2)") _);
        asm!("", out("mm0") _);
        asm!("{}", in(x87_reg) foo);
        //~^ ERROR register class `x87_reg` can only be used as a clobber, not as an input or output
        //~| ERROR `i32` cannot be used with this register class
        asm!("{}", in(mmx_reg) foo);
        //~^ ERROR register class `mmx_reg` can only be used as a clobber, not as an input or output
        //~| ERROR `i32` cannot be used with this register class
        asm!("{}", out(x87_reg) _);
        //~^ ERROR register class `x87_reg` can only be used as a clobber, not as an input or output
        asm!("{}", out(mmx_reg) _);
        //~^ ERROR register class `mmx_reg` can only be used as a clobber, not as an input or output

        // Explicit register conflicts
        // (except in/lateout which don't conflict)

        asm!("", in("eax") foo, in("al") bar);
        //~^ ERROR register `al` conflicts with register `eax`
        //~| ERROR `i32` cannot be used with this register class
        asm!("", in("rax") foo, out("rax") bar);
        //~^ ERROR register `rax` conflicts with register `rax`
        asm!("", in("al") foo, lateout("al") bar);
        //~^ ERROR `i32` cannot be used with this register class
        //~| ERROR `i32` cannot be used with this register class
        asm!("", in("xmm0") foo, in("ymm0") bar);
        //~^ ERROR register `ymm0` conflicts with register `xmm0`
        asm!("", in("xmm0") foo, out("ymm0") bar);
        //~^ ERROR register `ymm0` conflicts with register `xmm0`
        asm!("", in("xmm0") foo, lateout("ymm0") bar);
    }
}
