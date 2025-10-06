//@ only-x86_64
//@ run-pass
//@ needs-asm-support
//@ ignore-backends: gcc

#![deny(unreachable_code)]
#![feature(asm_goto_with_outputs)]

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

fn goto_out_jump_noreturn() {
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
            },
            options(noreturn)
        );
        assert!(value);
    }
}

// asm goto with outputs cause miscompilation in LLVM when multiple outputs are present.
// The code sample below is adapted from https://github.com/llvm/llvm-project/issues/74483
// and does not work with `-C opt-level=0`
#[expect(unused)]
fn goto_multi_out() {
    #[inline(never)]
    #[allow(unused)]
    fn goto_multi_out(a: usize, b: usize) -> usize {
        let mut x: usize;
        let mut y: usize;
        let mut z: usize;
        unsafe {
            core::arch::asm!(
                "mov {x}, {a}",
                "test {a}, {a}",
                "jnz {label1}",
                "/* {y} {z} {b} {label2} */",
                x = out(reg) x,
                y = out(reg) y,
                z = out(reg) z,
                a = in(reg) a,
                b = in(reg) b,
                label1 = label { return x },
                label2 = label { return 1 },
            );
            0
        }
    }

    assert_eq!(goto_multi_out(11, 22), 11);
}

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
    goto_out_fallthrough();
    goto_out_jump();
    goto_out_jump_noreturn();
    // goto_multi_out();
    goto_noreturn();
    goto_noreturn_diverge();
}
