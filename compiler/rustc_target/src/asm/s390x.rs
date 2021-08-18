use super::{InlineAsmArch, InlineAsmType};
use rustc_macros::HashStable_Generic;
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
    ) -> &'static [(InlineAsmType, Option<&'static str>)] {
        match (self, arch) {
            (Self::reg, _) => types! { _: I8, I16, I32; },
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
        r11: reg = ["r11"],
        r12: reg = ["r12"],
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
        #error = ["r13"] =>
            "The base pointer cannot be used as an operand for inline asm",
        #error = ["r15"] =>
            "The stack pointer cannot be used as an operand for inline asm",
        #error = ["a0"] =>
            "This pointer is reserved on s390x and cannot be used as an operand for inline asm",
        #error = ["a1"] =>
            "This pointer is reserved on z/Arch and cannot be used as an operand for inline asm",
        #error = ["c0"] =>
            "c0 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c1"] =>
            "c1 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c2"] =>
            "c2 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c3"] =>
            "c3 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c4"] =>
            "c4 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c5"] =>
            "c5 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c6"] =>
            "c6 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c7"] =>
            "c7 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c8"] =>
            "c8 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c9"] =>
            "c9 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c10"] =>
            "c10 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c11"] =>
            "c11 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c12"] =>
            "c12 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c13"] =>
            "c13 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c14"] =>
            "c14 is reserved by the kernel and cannot be used as an operand for inline asm",
        #error = ["c15"] =>
            "c15 is reserved by the kernel and cannot be used as an operand for inline asm",
	    #error = ["a2"] =>
            "a2 is not supported by LLVM and cannot be used as an operand for inline asm",
        #error = ["a3"] =>
            "a3 is not supported by LLVM and cannot be used as an operand for inline asm",
        #error = ["a4"] =>
            "a4 is not supported by LLVM and cannot be used as an operand for inline asm",
        #error = ["a5"] =>
            "a5 is not supported by LLVM and cannot be used as an operand for inline asm",
        #error = ["a6"] =>
            "a6 is not supported by LLVM and cannot be used as an operand for inline asm",
        #error = ["a7"] =>
            "a7 is not supported by LLVM and cannot be used as an operand for inline asm",
        #error = ["a8"] =>
            "a8 is not supported by LLVM and cannot be used as an operand for inline asm",
        #error = ["a9"] =>
            "a9 is not supported by LLVM and cannot be used as an operand for inline asm",
        #error = ["a10"] =>
            "a10 is not supported by LLVM and cannot be used as an operand for inline asm",
        #error = ["a11"] =>
            "a11 is not supported by LLVM and cannot be used as an operand for inline asm",
        #error = ["a12"] =>
            "a12 is not supported by LLVM and cannot be used as an operand for inline asm",
        #error = ["a13"] =>
            "a13 is not supported by LLVM and cannot be used as an operand for inline asm",
        #error = ["a14"] =>
            "a14 is not supported by LLVM and cannot be used as an operand for inline asm",
        #error = ["a15"] =>
            "a15 is not supported by LLVM and cannot be used as an operand for inline asm",
    }
}

impl S390xInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        out.write_str(self.name())
    }
}
