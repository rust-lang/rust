// only-aarch64
// compile-flags: -C target-feature=+fp

#![feature(asm, asm_const, asm_sym)]

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
        asm!("{:r}", in(vreg) foo);
        //~^ ERROR invalid asm template modifier for this register class
        asm!("{:r}", in(vreg_low16) foo);
        //~^ ERROR invalid asm template modifier for this register class
        asm!("{:a}", const 0);
        //~^ ERROR asm template modifiers are not allowed for `const` arguments
        asm!("{:a}", sym main);
        //~^ ERROR asm template modifiers are not allowed for `sym` arguments
        asm!("", in("x29") foo);
        //~^ ERROR invalid register `x29`: the frame pointer cannot be used as an operand
        asm!("", in("sp") foo);
        //~^ ERROR invalid register `sp`: the stack pointer cannot be used as an operand
        asm!("", in("xzr") foo);
        //~^ ERROR invalid register `xzr`: the zero register cannot be used as an operand
        asm!("", in("x18") foo);
        //~^ ERROR invalid register `x18`: x18 is used as a reserved register on some targets and cannot be used as an operand for inline asm
        asm!("", in("x19") foo);
        //~^ ERROR invalid register `x19`: x19 is used internally by LLVM and cannot be used as an operand for inline asm

        asm!("", in("p0") foo);
        //~^ ERROR register class `preg` can only be used as a clobber, not as an input or output
        asm!("", out("p0") _);
        asm!("{}", in(preg) foo);
        //~^ ERROR register class `preg` can only be used as a clobber, not as an input or output
        asm!("{}", out(preg) _);
        //~^ ERROR register class `preg` can only be used as a clobber, not as an input or output

        // Explicit register conflicts
        // (except in/lateout which don't conflict)

        asm!("", in("x0") foo, in("w0") bar);
        //~^ ERROR register `x0` conflicts with register `x0`
        asm!("", in("x0") foo, out("x0") bar);
        //~^ ERROR register `x0` conflicts with register `x0`
        asm!("", in("w0") foo, lateout("w0") bar);
        asm!("", in("v0") foo, in("q0") bar);
        //~^ ERROR register `v0` conflicts with register `v0`
        asm!("", in("v0") foo, out("q0") bar);
        //~^ ERROR register `v0` conflicts with register `v0`
        asm!("", in("v0") foo, lateout("q0") bar);
    }
}
