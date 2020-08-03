#![feature(llvm_asm)]

enum Empty {}

fn empty() -> Option<Empty> {
    None
}

// EMIT_MIR unreachable_asm_2.main.UnreachablePropagation.diff
fn main() {
    if let Some(_x) = empty() {
        let mut _y;

        if true {
            // asm instruction stops unreachable propagation to block bb3.
            unsafe { llvm_asm!("NOP"); }
            _y = 21;
        } else {
            // asm instruction stops unreachable propagation to block bb3.
            unsafe { llvm_asm!("NOP"); }
            _y = 42;
        }

        match _x { }
    }
}
