#![feature(llvm_asm)]

enum Empty {}

fn empty() -> Option<Empty> {
    None
}

// EMIT_MIR rustc.main.UnreachablePropagation.diff
fn main() {
    if let Some(_x) = empty() {
        let mut _y;

        if true {
            _y = 21;
        } else {
            _y = 42;
        }

        // asm instruction stops unreachable propagation to if else blocks bb4 and bb5.
        unsafe { llvm_asm!("NOP"); }
        match _x { }
    }
}
