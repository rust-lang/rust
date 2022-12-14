use super::{InlineAsmArch, InlineAsmType};
use rustc_macros::HashStable_Generic;
use rustc_span::Symbol;
use std::fmt;

def_reg_class! {
    S390x S390xInlineAsmRegClass {
        reg,
        freg,
    }
}

impl S390xInlineAsmRegClass {
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
            (Self::reg, _) => types! { _: I8, I16, I32, I64; },
            (Self::freg, _) => types! { _: F32, F64; },
        }
    }
}

def_regs! {
    S390x S390xInlineAsmReg S390xInlineAsmRegClass {
        r0: reg = ["r0"],
        r1: reg = ["r1"],
        r2: reg = ["r2"],
        r3: reg = ["r3"],
        r4: reg = ["r4"],
        r5: reg = ["r5"],
        r6: reg = ["r6"],
        r7: reg = ["r7"],
        r8: reg = ["r8"],
        r9: reg = ["r9"],
        r10: reg = ["r10"],
        r12: reg = ["r12"],
        r13: reg = ["r13"],
        r14: reg = ["r14"],
        f0: freg = ["f0"],
        f1: freg = ["f1"],
        f2: freg = ["f2"],
        f3: freg = ["f3"],
        f4: freg = ["f4"],
        f5: freg = ["f5"],
        f6: freg = ["f6"],
        f7: freg = ["f7"],
        f8: freg = ["f8"],
        f9: freg = ["f9"],
        f10: freg = ["f10"],
        f11: freg = ["f11"],
        f12: freg = ["f12"],
        f13: freg = ["f13"],
        f14: freg = ["f14"],
        f15: freg = ["f15"],
        #error = ["r11"] =>
            "The frame pointer cannot be used as an operand for inline asm",
        #error = ["r15"] =>
            "The stack pointer cannot be used as an operand for inline asm",
        #error = [
            "c0", "c1", "c2", "c3",
            "c4", "c5", "c6", "c7",
            "c8", "c9", "c10", "c11",
            "c12", "c13", "c14", "c15"
        ] =>
            "control registers are reserved by the kernel and cannot be used as operands for inline asm",
        #error = [
            "a0", "a1", "a2", "a3",
            "a4", "a5", "a6", "a7",
            "a8", "a9", "a10", "a11",
            "a12", "a13", "a14", "a15"
        ] =>
            "access registers are not supported and cannot be used as operands for inline asm",
    }
}

impl S390xInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        write!(out, "%{}", self.name())
    }
}
