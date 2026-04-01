#![feature(linkage)]
#![crate_type = "dylib"]

use std::arch::naked_asm;

pub trait TraitWithConst {
    const COUNT: u32;
}

struct Test;

impl TraitWithConst for Test {
    const COUNT: u32 = 1;
}

#[no_mangle]
fn entry() {
    private_vanilla();
    private_naked();

    public_vanilla_generic::<Test>();
    public_naked_generic::<Test>();
}

extern "C" fn private_vanilla() -> u32 {
    42
}

#[unsafe(naked)]
extern "C" fn private_naked() -> u32 {
    naked_asm!("mov rax, 42", "ret")
}

#[no_mangle]
pub extern "C" fn public_vanilla() -> u32 {
    42
}

#[unsafe(naked)]
#[no_mangle]
pub extern "C" fn public_naked_nongeneric() -> u32 {
    naked_asm!("mov rax, 42", "ret")
}

pub extern "C" fn public_vanilla_generic<T: TraitWithConst>() -> u32 {
    T::COUNT
}

#[unsafe(naked)]
pub extern "C" fn public_naked_generic<T: TraitWithConst>() -> u32 {
    naked_asm!("mov rax, {}", "ret", const T::COUNT)
}

#[linkage = "external"]
extern "C" fn vanilla_external_linkage() -> u32 {
    42
}

#[unsafe(naked)]
#[linkage = "external"]
extern "C" fn naked_external_linkage() -> u32 {
    naked_asm!("mov rax, 42", "ret")
}

#[cfg(not(windows))]
#[linkage = "weak"]
extern "C" fn vanilla_weak_linkage() -> u32 {
    42
}

#[unsafe(naked)]
#[cfg(not(windows))]
#[linkage = "weak"]
extern "C" fn naked_weak_linkage() -> u32 {
    naked_asm!("mov rax, 42", "ret")
}

// functions that are declared in an `extern "C"` block are currently not exported
// this maybe should change in the future, this is just tracking the current behavior
// reported in https://github.com/rust-lang/rust/issues/128071
std::arch::global_asm! {
    ".globl function_defined_in_global_asm",
    "function_defined_in_global_asm:",
    "ret",
}

extern "C" {
    pub fn function_defined_in_global_asm();
}
