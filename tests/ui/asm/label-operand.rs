//@ run-pass
//@ reference: asm.operand-type.supported-operands.label
//@ revisions: aarch64 arm arm64ec riscv32 riscv64 x86 x86_64
//@ needs-asm-support
//@[aarch64] only-aarch64
//@[arm64ec] only-arm64ec
//@[arm] only-arm
//@[riscv32] only-riscv32
//@[riscv64] only-riscv64
//@[x86] only-x86
//@[x86_64] only-x86_64

#[cfg(any(aarch64, arm, arm64ec))]
fn make_true(value: &mut bool) {
    unsafe {
        core::arch::asm!(
            "b {}",
            label {
                *value = true;
            }
        );
    }
}

#[cfg(any(riscv32, riscv64))]
fn make_true(value: &mut bool) {
    unsafe {
        core::arch::asm!(
            "j {}",
            label {
                *value = true;
            }
        );
    }
}

#[cfg(any(x86, x86_64))]
fn make_true(value: &mut bool) {
    unsafe {
        core::arch::asm!(
            "jmp {}",
            label {
                *value = true;
            }
        );
    }
}

fn main() {
    let mut value = false;
    make_true(&mut value);
    assert!(value);
}
