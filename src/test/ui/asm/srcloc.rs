// no-system-llvm
// only-x86_64
// build-fail

#![feature(asm)]

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
    }
}
