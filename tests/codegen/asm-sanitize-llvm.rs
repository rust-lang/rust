// FIXME(nagisa): remove the flags below once all targets support `asm!`.
// compile-flags: --target x86_64-unknown-linux-gnu -Copt-level=0
// needs-llvm-components: x86

// Verify we sanitize the special tokens for the LLVM inline-assembly, ensuring people won't
// inadvertently rely on the LLVM-specific syntax and features.
#![no_core]
#![feature(no_core, lang_items, rustc_attrs)]
#![crate_type = "rlib"]
#![allow(named_asm_labels)]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

pub unsafe fn we_escape_dollar_signs() {
    // CHECK: call void asm sideeffect alignstack inteldialect "banana$$:"
    asm!(
        r"banana$:",
    )
}

pub unsafe fn we_escape_escapes_too() {
    // CHECK: call void asm sideeffect alignstack inteldialect "banana\{{(\\|5C)}}36:"
    asm!(
        r"banana\36:",
    )
}
