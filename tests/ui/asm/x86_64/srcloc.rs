//@ only-x86_64
//@ build-fail
//@ compile-flags: -Ccodegen-units=1
//@ ignore-backends: gcc

use std::arch::asm;

// Checks that inline asm errors are mapped to the correct line in the source code.

fn main() {
    unsafe {
        asm!("invalid_instruction");
        //~^ ERROR: invalid instruction mnemonic 'invalid_instruction'

        asm!("
            invalid_instruction
        ");
        //~^^ ERROR: invalid instruction mnemonic 'invalid_instruction'

        asm!(r#"
            invalid_instruction
        "#);
        //~^^ ERROR: invalid instruction mnemonic 'invalid_instruction'

        asm!("
            mov eax, eax
            invalid_instruction
            mov eax, eax
        ");
        //~^^^ ERROR: invalid instruction mnemonic 'invalid_instruction'

        asm!(r#"
            mov eax, eax
            invalid_instruction
            mov eax, eax
        "#);
        //~^^^ ERROR: invalid instruction mnemonic 'invalid_instruction'

        asm!(concat!("invalid", "_", "instruction"));
        //~^ ERROR: invalid instruction mnemonic 'invalid_instruction'

        asm!("movaps %xmm3, (%esi, 2)", options(att_syntax));
        //~^ WARN: scale factor without index register is ignored

        asm!(
            "invalid_instruction",
        );
        //~^^ ERROR: invalid instruction mnemonic 'invalid_instruction'

        asm!(
            "mov eax, eax",
            "invalid_instruction",
            "mov eax, eax",
        );
        //~^^^ ERROR: invalid instruction mnemonic 'invalid_instruction'

        asm!(
            "mov eax, eax\n",
            "invalid_instruction",
            "mov eax, eax",
        );
        //~^^^ ERROR: invalid instruction mnemonic 'invalid_instruction'

        asm!(
            "mov eax, eax",
            concat!("invalid", "_", "instruction"),
            "mov eax, eax",
        );
        //~^^^ ERROR: invalid instruction mnemonic 'invalid_instruction'

        asm!(
            concat!("mov eax", ", ", "eax"),
            concat!("invalid", "_", "instruction"),
            concat!("mov eax", ", ", "eax"),
        );
        //~^^^ ERROR: invalid instruction mnemonic 'invalid_instruction'

        // Make sure template strings get separated
        asm!(
            "invalid_instruction1",
            "invalid_instruction2",
        );
        //~^^^ ERROR: invalid instruction mnemonic 'invalid_instruction1'
        //~^^^ ERROR: invalid instruction mnemonic 'invalid_instruction2'

        asm!(
            concat!(
                "invalid", "_", "instruction1", "\n",
                "invalid", "_", "instruction2",
            ),
        );
        //~^^^^^ ERROR: invalid instruction mnemonic 'invalid_instruction1'
        //~^^^^^^ ERROR: invalid instruction mnemonic 'invalid_instruction2'

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
        //~^^^^^^^^^ ERROR: invalid instruction mnemonic 'invalid_instruction1'
        //~^^^^^^^^^^ ERROR: invalid instruction mnemonic 'invalid_instruction2'
        //~^^^^^^^ ERROR: invalid instruction mnemonic 'invalid_instruction3'
        //~^^^^^^^^ ERROR: invalid instruction mnemonic 'invalid_instruction4'

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
        //~^^^^^^^^^ ERROR: invalid instruction mnemonic 'invalid_instruction1'
        //~^^^^^^^^^^ ERROR: invalid instruction mnemonic 'invalid_instruction2'
        //~^^^^^^^ ERROR: invalid instruction mnemonic 'invalid_instruction3'
        //~^^^^^^^^ ERROR: invalid instruction mnemonic 'invalid_instruction4'

        asm!(
            "",
            "\n",
            "invalid_instruction"
        );
        //~^^ ERROR: invalid instruction mnemonic 'invalid_instruction'
    }
}
