use super::{InlineAsmArch, InlineAsmType};
use rustc_macros::HashStable_Generic;
use rustc_span::Symbol;
use std::fmt;

def_reg_class! {
    LoongArch LoongArchInlineAsmRegClass {
        reg,
        freg,
    }
}

impl LoongArchInlineAsmRegClass {
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
        _arch: InlineAsmArch,
    ) -> &'static [(InlineAsmType, Option<Symbol>)] {
        match self {
            Self::reg => types! { _: I8, I16, I32, I64, F32, F64; },
            Self::freg => types! { _: F32, F64; },
        }
    }
}

// The reserved registers are taken from <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/LoongArch/LoongArchRegisterInfo.cpp#79>
def_regs! {
    LoongArch LoongArchInlineAsmReg LoongArchInlineAsmRegClass {
        r1: reg = ["$r1","$ra"],
        r4: reg = ["$r4","$a0"],
        r5: reg = ["$r5","$a1"],
        r6: reg = ["$r6","$a2"],
        r7: reg = ["$r7","$a3"],
        r8: reg = ["$r8","$a4"],
        r9: reg = ["$r9","$a5"],
        r10: reg = ["$r10","$a6"],
        r11: reg = ["$r11","$a7"],
        r12: reg = ["$r12","$t0"],
        r13: reg = ["$r13","$t1"],
        r14: reg = ["$r14","$t2"],
        r15: reg = ["$r15","$t3"],
        r16: reg = ["$r16","$t4"],
        r17: reg = ["$r17","$t5"],
        r18: reg = ["$r18","$t6"],
        r19: reg = ["$r19","$t7"],
        r20: reg = ["$r20","$t8"],
        r23: reg = ["$r23","$s0"],
        r24: reg = ["$r24","$s1"],
        r25: reg = ["$r25","$s2"],
        r26: reg = ["$r26","$s3"],
        r27: reg = ["$r27","$s4"],
        r28: reg = ["$r28","$s5"],
        r29: reg = ["$r29","$s6"],
        r30: reg = ["$r30","$s7"],
        f0: freg = ["$f0","$fa0"],
        f1: freg = ["$f1","$fa1"],
        f2: freg = ["$f2","$fa2"],
        f3: freg = ["$f3","$fa3"],
        f4: freg = ["$f4","$fa4"],
        f5: freg = ["$f5","$fa5"],
        f6: freg = ["$f6","$fa6"],
        f7: freg = ["$f7","$fa7"],
        f8: freg = ["$f8","$ft0"],
        f9: freg = ["$f9","$ft1"],
        f10: freg = ["$f10","$ft2"],
        f11: freg = ["$f11","$ft3"],
        f12: freg = ["$f12","$ft4"],
        f13: freg = ["$f13","$ft5"],
        f14: freg = ["$f14","$ft6"],
        f15: freg = ["$f15","$ft7"],
        f16: freg = ["$f16","$ft8"],
        f17: freg = ["$f17","$ft9"],
        f18: freg = ["$f18","$ft10"],
        f19: freg = ["$f19","$ft11"],
        f20: freg = ["$f20","$ft12"],
        f21: freg = ["$f21","$ft13"],
        f22: freg = ["$f22","$ft14"],
        f23: freg = ["$f23","$ft15"],
        f24: freg = ["$f24","$fs0"],
        f25: freg = ["$f25","$fs1"],
        f26: freg = ["$f26","$fs2"],
        f27: freg = ["$f27","$fs3"],
        f28: freg = ["$f28","$fs4"],
        f29: freg = ["$f29","$fs5"],
        f30: freg = ["$f30","$fs6"],
        f31: freg = ["$f31","$fs7"],
        #error = ["$r0","$zero"] =>
            "constant zero cannot be used as an operand for inline asm",
        #error = ["$r2","$tp"] =>
            "reserved for TLS",
        #error = ["$r3","$sp"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["$r21"] =>
            "reserved by the ABI",
        #error = ["$r22","$fp"] =>
            "the frame pointer cannot be used as an operand for inline asm",
        #error = ["$r31","$s8"] =>
            "$r31 is used internally by LLVM and cannot be used as an operand for inline asm",
    }
}

impl LoongArchInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        out.write_str(self.name())
    }
}
