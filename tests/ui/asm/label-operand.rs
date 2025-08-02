//@ run-pass
//@ needs-asm-support
//@ reference: asm.operand-type.supported-operands.label

use std::arch::asm;

#[cfg(any(target_arch = "arm", target_arch = "aarch64", target_arch = "arm64ec"))]
fn make_true(value: &mut bool) {
    unsafe {
        asm!(
            "b {}",
            label {
                *value = true;
            }
        );
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
fn make_true(value: &mut bool) {
    unsafe {
        asm!(
            "j {}",
            label {
                *value = true;
            }
        );
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn make_true(value: &mut bool) {
    unsafe {
        asm!(
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
