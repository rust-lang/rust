// Check that `cfg` and `cfg_attr` work as expected.
//
//@ revisions: reva revb
//@ only-x86_64
//@ run-pass
#![feature(asm_cfg, cfg_select)]

use std::arch::{asm, naked_asm};

#[unsafe(naked)]
extern "C" fn ignore_const_operand() -> u64 {
    naked_asm!(
        "mov rax, 5",
        #[cfg(revb)]
        "mov rax, {a}",
        "ret",
        #[cfg(revb)]
        a = const 10,
    )
}

#[unsafe(naked)]
extern "C" fn ignore_const_operand_cfg_attr() -> u64 {
    naked_asm!(
        "mov rax, 5",
        #[cfg_attr(true, cfg(revb))]
        "mov rax, {a}",
        "ret",
        #[cfg_attr(true, cfg(revb))]
        a = const 10,
    )
}

#[unsafe(naked)]
extern "C" fn const_operand() -> u64 {
    naked_asm!(
        "mov rax, {a}",
        "ret",
        #[cfg(reva)]
        a = const 5,
        #[cfg(revb)]
        a = const 10,
    )
}

fn options() {
    // Without the cfg, this throws an error that the `att_syntax` option is provided twice.
    unsafe {
        asm!(
            "nop",
            #[cfg(false)]
            options(att_syntax),
            options(att_syntax)
        )
    }
}

fn clobber_abi() {
    // Without the cfg, this throws an error that the "C" abi is provided twice.
    unsafe {
        asm!(
            "nop",
            #[cfg(false)]
            clobber_abi("C"),
            clobber_abi("C"),
        );
    }
}

#[unsafe(naked)]
extern "C" fn first_template() -> u64 {
    naked_asm!(
        #[cfg(reva)]
        "mov rax, 5",
        #[cfg(revb)]
        "mov rax, 10",
        "ret",
    )
}

#[unsafe(naked)]
extern "C" fn true_and_false() -> u64 {
    naked_asm!(
        "mov rax, 5",
        #[cfg(true)]
        #[cfg(false)]
        "mov rax, 10",
        "ret",
    )
}

#[unsafe(naked)]
extern "C" fn false_and_true() -> u64 {
    naked_asm!(
        "mov rax, 5",
        #[cfg(false)]
        #[cfg(true)]
        "mov rax, 10",
        "ret",
    )
}

pub fn main() {
    std::cfg_select! {
        reva => {
            assert_eq!(const_operand(), 5);
            assert_eq!(ignore_const_operand_cfg_attr(), 5);
            assert_eq!(ignore_const_operand(), 5);
            assert_eq!(first_template(), 5);

        }
        revb => {
            assert_eq!(const_operand(), 10);
            assert_eq!(ignore_const_operand_cfg_attr(), 10);
            assert_eq!(ignore_const_operand(), 10);
            assert_eq!(first_template(), 10);

        }
    }
    options();
    clobber_abi();

    assert_eq!(true_and_false(), 5);
    assert_eq!(false_and_true(), 5);
}
