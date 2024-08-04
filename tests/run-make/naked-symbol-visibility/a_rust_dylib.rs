#![feature(naked_functions, asm_const, linkage)]
#![crate_type = "dylib"]

use std::arch::asm;

pub trait TraitWithConst {
    const COUNT: u32;
}

struct Test;

impl TraitWithConst for Test {
    const COUNT: u32 = 1;
}

#[no_mangle]
fn entry() {
    private_vanilla_rust_function_from_rust_dylib();
    private_naked_rust_function_from_rust_dylib();

    public_vanilla_generic_function_from_rust_dylib::<Test>();
    public_naked_generic_function_from_rust_dylib::<Test>();
}

extern "C" fn private_vanilla_rust_function_from_rust_dylib() -> u32 {
    42
}

#[no_mangle]
pub extern "C" fn public_vanilla_rust_function_from_rust_dylib() -> u32 {
    42
}

pub extern "C" fn public_vanilla_generic_function_from_rust_dylib<T: TraitWithConst>() -> u32 {
    T::COUNT
}

#[linkage = "weak"]
extern "C" fn vanilla_weak_linkage() -> u32 {
    42
}

#[linkage = "external"]
extern "C" fn vanilla_external_linkage() -> u32 {
    42
}

#[naked]
extern "C" fn private_naked_rust_function_from_rust_dylib() -> u32 {
    unsafe { asm!("mov rax, 42", "ret", options(noreturn)) }
}

#[naked]
#[no_mangle]
pub extern "C" fn public_naked_rust_function_from_rust_dylib() -> u32 {
    unsafe { asm!("mov rax, 42", "ret", options(noreturn)) }
}

#[naked]
pub extern "C" fn public_naked_generic_function_from_rust_dylib<T: TraitWithConst>() -> u32 {
    unsafe { asm!("mov rax, {}", "ret", const T::COUNT, options(noreturn)) }
}

#[naked]
#[linkage = "weak"]
extern "C" fn naked_weak_linkage() -> u32 {
    unsafe { asm!("mov rax, 42", "ret", options(noreturn)) }
}

#[naked]
#[linkage = "external"]
extern "C" fn naked_external_linkage() -> u32 {
    unsafe { asm!("mov rax, 42", "ret", options(noreturn)) }
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
