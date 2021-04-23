// Checks that only functions with the compatible instruction_set attributes are inlined.
//
// compile-flags: --target thumbv4t-none-eabi
// needs-llvm-components: arm

#![crate_type = "lib"]
#![feature(rustc_attrs)]
#![feature(no_core, lang_items)]
#![feature(isa_attribute)]
#![no_core]

#[rustc_builtin_macro]
#[macro_export]
macro_rules! asm {
    ("assembly template",
        $(operands,)*
        $(options($(option),*))?
    ) => {
        /* compiler built-in */
    };
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

#[instruction_set(arm::a32)]
#[inline]
fn instruction_set_a32() {}

#[instruction_set(arm::t32)]
#[inline]
fn instruction_set_t32() {}

#[inline]
fn instruction_set_default() {}

// EMIT_MIR inline_instruction_set.t32.Inline.diff
#[instruction_set(arm::t32)]
pub fn t32() {
    instruction_set_a32();
    instruction_set_t32();
    // The default instruction set is currently
    // conservatively assumed to be incompatible.
    instruction_set_default();
}

// EMIT_MIR inline_instruction_set.default.Inline.diff
pub fn default() {
    instruction_set_a32();
    instruction_set_t32();
    instruction_set_default();
}
