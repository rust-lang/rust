pub fn ceil(mut x: f64) -> f64 {
    unsafe {
        core::arch::asm!(
            "fld qword ptr [{x}]",
            // Save the FPU control word, using `x` as scratch space.
            "fstcw [{x}]",
            // Set rounding control to 0b10 (+∞).
            "mov word ptr [{x} + 2], 0x0b7f",
            "fldcw [{x} + 2]",
            // Round.
            "frndint",
            // Restore FPU control word.
            "fldcw [{x}]",
            // Save rounded value to memory.
            "fstp qword ptr [{x}]",
            x = in(reg) &mut x,
            // All the x87 FPU stack is used, all registers must be clobbered
            out("st(0)") _, out("st(1)") _,
            out("st(2)") _, out("st(3)") _,
            out("st(4)") _, out("st(5)") _,
            out("st(6)") _, out("st(7)") _,
            options(nostack),
        );
    }
    x
}

pub fn floor(mut x: f64) -> f64 {
    unsafe {
        core::arch::asm!(
            "fld qword ptr [{x}]",
            // Save the FPU control word, using `x` as scratch space.
            "fstcw [{x}]",
            // Set rounding control to 0b01 (-∞).
            "mov word ptr [{x} + 2], 0x077f",
            "fldcw [{x} + 2]",
            // Round.
            "frndint",
            // Restore FPU control word.
            "fldcw [{x}]",
            // Save rounded value to memory.
            "fstp qword ptr [{x}]",
            x = in(reg) &mut x,
            // All the x87 FPU stack is used, all registers must be clobbered
            out("st(0)") _, out("st(1)") _,
            out("st(2)") _, out("st(3)") _,
            out("st(4)") _, out("st(5)") _,
            out("st(6)") _, out("st(7)") _,
            options(nostack),
        );
    }
    x
}

/// Note that this respects rounding mode. Because it is UB to have a non-default rounding
/// mode in Rust, this acts as roundeven.
pub fn rint(mut x: f64) -> f64 {
    unsafe {
        core::arch::asm!(
            "fld qword ptr [{x}]",
            "frndint",
            "fstp qword ptr [{x}]",
            x = in(reg) &mut x,
            // All the x87 FPU stack is used, all registers must be clobbered
            out("st(0)") _, out("st(1)") _,
            out("st(2)") _, out("st(3)") _,
            out("st(4)") _, out("st(5)") _,
            out("st(6)") _, out("st(7)") _,
            options(nostack),
        );
    }
    x
}

/* FIXME(msrv): after 1.82, the below can be used to compute control words using `asm_const`:

#[derive(Clone, Copy, Debug, PartialEq)]
enum Precision {
    Single,
    Double,
    Extended,
}

/// See: Intel® 64 and IA-32 Architectures Software Developer's Manual Volume 1:
/// Basic Architecture, section 8.1.5 x87 FPU Control Word.
const fn make_fpcw(round: Round, prec: Precision) -> u16 {
    let exceptions = 0b111111; // Disable all 6 exceptions
    let misc = 0b1000000; // reserved field usually set by default
    let pc = match prec {
        Precision::Single => 0b00,
        Precision::Double => 0b10,
        Precision::Extended => 0b11,
    };
    let rc = match round {
        Round::Nearest => 0b00,
        Round::Negative => 0b01,
        Round::Positive => 0b10,
        Round::Zero => 0b11,
    };
    (rc << 10) | (pc << 8) | misc | exceptions
}

*/
