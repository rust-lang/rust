//@ only-x86_64
//@ run-pass
//@ needs-asm-support
//@ revisions: mirunsafeck thirunsafeck
//@ [thirunsafeck]compile-flags: -Z thir-unsafeck

#![deny(unreachable_code)]
#![feature(asm_goto)]

use std::arch::asm;

fn goto_fallthough() {
    unsafe {
        asm!(
            "/* {} */",
            label {
                unreachable!();
            }
        )
    }
}

fn goto_jump() {
    unsafe {
        let mut value = false;
        asm!(
            "jmp {}",
            label {
                value = true;
            }
        );
        assert!(value);
    }
}

// asm goto with outputs cause miscompilation in LLVM. UB can be triggered
// when outputs are used inside the label block when optimisation is enabled.
// See: https://github.com/llvm/llvm-project/issues/74483
/*
fn goto_out_fallthrough() {
    unsafe {
        let mut out: usize;
        asm!(
            "lea {}, [{} + 1]",
            "/* {} */",
            out(reg) out,
            in(reg) 0x12345678usize,
            label {
                unreachable!();
            }
        );
        assert_eq!(out, 0x12345679);
    }
}

fn goto_out_jump() {
    unsafe {
        let mut value = false;
        let mut out: usize;
        asm!(
            "lea {}, [{} + 1]",
            "jmp {}",
            out(reg) out,
            in(reg) 0x12345678usize,
            label {
                value = true;
                assert_eq!(out, 0x12345679);
            }
        );
        assert!(value);
    }
}
*/

fn goto_noreturn() {
    unsafe {
        let a;
        asm!(
            "jmp {}",
            label {
                a = 1;
            },
            options(noreturn)
        );
        assert_eq!(a, 1);
    }
}

#[warn(unreachable_code)]
fn goto_noreturn_diverge() {
    unsafe {
        asm!(
            "jmp {}",
            label {
                return;
            },
            options(noreturn)
        );
        unreachable!();
        //~^ WARN unreachable statement
    }
}

fn main() {
    goto_fallthough();
    goto_jump();
    // goto_out_fallthrough();
    // goto_out_jump();
    goto_noreturn();
    goto_noreturn_diverge();
}
