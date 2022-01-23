use super::{InlineAsmArch, InlineAsmType};
use rustc_macros::HashStable_Generic;
use rustc_span::Symbol;
use std::fmt;

def_reg_class! {
    Msp430 Msp430InlineAsmRegClass {
        reg,
    }
}

impl Msp430InlineAsmRegClass {
    pub fn valid_modifiers(self, _arch: super::InlineAsmArch) -> &'static [char] {
        &[]
    }

    pub fn suggest_class(self, _arch: InlineAsmArch, _ty: InlineAsmType) -> Option<Self> {
        None
    }

    pub fn suggest_modifier(
        self,
        _arch: InlineAsmArch,
        _ty: InlineAsmType,
    ) -> Option<(char, &'static str)> {
        None
    }

    pub fn default_modifier(self, _arch: InlineAsmArch) -> Option<(char, &'static str)> {
        None
    }

    pub fn supported_types(
        self,
        arch: InlineAsmArch,
    ) -> &'static [(InlineAsmType, Option<Symbol>)] {
        match (self, arch) {
            (Self::reg, _) => types! { _: I8, I16; },
        }
    }
}

// The reserved registers are taken from:
// https://github.com/llvm/llvm-project/blob/36cb29cbbe1b22dcd298ad65e1fabe899b7d7249/llvm/lib/Target/MSP430/MSP430RegisterInfo.cpp#L73.
def_regs! {
    Msp430 Msp430InlineAsmReg Msp430InlineAsmRegClass {
        r5: reg = ["r5"],
        r6: reg = ["r6"],
        r7: reg = ["r7"],
        r8: reg = ["r8"],
        r9: reg = ["r9"],
        r10: reg = ["r10"],
        r11: reg = ["r11"],
        r12: reg = ["r12"],
        r13: reg = ["r13"],
        r14: reg = ["r14"],
        r15: reg = ["r15"],

        #error = ["r0", "pc"] =>
            "the program counter cannot be used as an operand for inline asm",
        #error = ["r1", "sp"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["r2", "sr"] =>
            "the status register cannot be used as an operand for inline asm",
        #error = ["r3", "cg"] =>
            "the constant generator cannot be used as an operand for inline asm",
        #error = ["r4", "fp"] =>
            "the frame pointer cannot be used as an operand for inline asm",
    }
}

impl Msp430InlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        out.write_str(self.name())
    }
}
