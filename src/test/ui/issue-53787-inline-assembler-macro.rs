// Regression test for Issue #53787: Fix ICE when creating a label in inline assembler with macros.

#![feature(asm)]

macro_rules! fake_jump {
    ($id:expr) => {
        unsafe {
            asm!(
            "
            jmp $0
            lea eax, [ebx]
            xor eax, 0xDEADBEEF
            retn
            $0:
            "::"0"($id)::"volatile", "intel");
        }
    };
}

fn main() {
    fake_jump!("FirstFunc"); //~ ERROR invalid value for constraint in inline assembly
    println!("Hello, world!");
}
