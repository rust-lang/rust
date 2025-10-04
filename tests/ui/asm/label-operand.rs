//@ run-pass
//@ reference: asm.operand-type.supported-operands.label
//@ revisions: arm arm64 riscv32 riscv64 x86 other
//@ needs-asm-support
//@[riscv64] only-riscv64
//@[riscv32] only-riscv32
//@[arm] only-arm
//@[arm64] only-aarch64
//@[x86] only-x86

#[cfg(any(arm, arm64))]
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

#[cfg(x86)]
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

// fallback
#[cfg(other)]
fn make_true(value: &mut bool) {
    *value = true;
}

fn main() {
    let mut value = false;
    make_true(&mut value);
    assert!(value);
}
