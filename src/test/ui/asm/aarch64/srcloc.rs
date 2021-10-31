// only-aarch64
// build-fail
// compile-flags: -Ccodegen-units=1
#![feature(asm)]

// Checks that inline asm errors are mapped to the correct line in the source code.

fn main() {
    unsafe {
        asm!("invalid_instruction");
        //~^ ERROR: unrecognized instruction mnemonic

        asm!("
            invalid_instruction
        ");
        //~^^ ERROR: unrecognized instruction mnemonic

        asm!(r#"
            invalid_instruction
        "#);
        //~^^ ERROR: unrecognized instruction mnemonic

        asm!("
            mov x0, x0
            invalid_instruction
            mov x0, x0
        ");
        //~^^^ ERROR: unrecognized instruction mnemonic

        asm!(r#"
            mov x0, x0
            invalid_instruction
            mov x0, x0
        "#);
        //~^^^ ERROR: unrecognized instruction mnemonic

        asm!(concat!("invalid", "_", "instruction"));
        //~^ ERROR: unrecognized instruction mnemonic

        asm!(
            "invalid_instruction",
        );
        //~^^ ERROR: unrecognized instruction mnemonic

        asm!(
            "mov x0, x0",
            "invalid_instruction",
            "mov x0, x0",
        );
        //~^^^ ERROR: unrecognized instruction mnemonic

        asm!(
            "mov x0, x0\n",
            "invalid_instruction",
            "mov x0, x0",
        );
        //~^^^ ERROR: unrecognized instruction mnemonic

        asm!(
            "mov x0, x0",
            concat!("invalid", "_", "instruction"),
            "mov x0, x0",
        );
        //~^^^ ERROR: unrecognized instruction mnemonic

        asm!(
            concat!("mov x0", ", ", "x0"),
            concat!("invalid", "_", "instruction"),
            concat!("mov x0", ", ", "x0"),
        );
        //~^^^ ERROR: unrecognized instruction mnemonic

        // Make sure template strings get separated
        asm!(
            "invalid_instruction1",
            "invalid_instruction2",
        );
        //~^^^ ERROR: unrecognized instruction mnemonic
        //~^^^ ERROR: unrecognized instruction mnemonic

        asm!(
            concat!(
                "invalid", "_", "instruction1", "\n",
                "invalid", "_", "instruction2",
            ),
        );
        //~^^^^^ ERROR: unrecognized instruction mnemonic
        //~^^^^^^ ERROR: unrecognized instruction mnemonic

        asm!(
            concat!(
                "invalid", "_", "instruction1", "\n",
                "invalid", "_", "instruction2",
            ),
            concat!(
                "invalid", "_", "instruction3", "\n",
                "invalid", "_", "instruction4",
            ),
        );
        //~^^^^^^^^^ ERROR: unrecognized instruction mnemonic
        //~^^^^^^^^^^ ERROR: unrecognized instruction mnemonic
        //~^^^^^^^ ERROR: unrecognized instruction mnemonic
        //~^^^^^^^^ ERROR: unrecognized instruction mnemonic

        asm!(
            concat!(
                "invalid", "_", "instruction1", "\n",
                "invalid", "_", "instruction2", "\n",
            ),
            concat!(
                "invalid", "_", "instruction3", "\n",
                "invalid", "_", "instruction4", "\n",
            ),
        );
        //~^^^^^^^^^ ERROR: unrecognized instruction mnemonic
        //~^^^^^^^^^^ ERROR: unrecognized instruction mnemonic
        //~^^^^^^^ ERROR: unrecognized instruction mnemonic
        //~^^^^^^^^ ERROR: unrecognized instruction mnemonic
    }
}
