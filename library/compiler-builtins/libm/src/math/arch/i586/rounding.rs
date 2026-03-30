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
