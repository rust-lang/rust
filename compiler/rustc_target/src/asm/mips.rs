use super::{InlineAsmArch, InlineAsmType};
use rustc_macros::HashStable_Generic;
use std::fmt;

def_reg_class! {
    Mips MipsInlineAsmRegClass {
        reg,
        freg,
    }
}

impl MipsInlineAsmRegClass {
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
    ) -> &'static [(InlineAsmType, Option<&'static str>)] {
        match self {
            Self::reg => types! { _: I8, I16, I32, F32; },
            Self::freg => types! { _: F32; },
        }
    }
}

// The reserved registers are somewhat taken from <https://git.io/JUR1k#L150>.
def_regs! {
    Mips MipsInlineAsmReg MipsInlineAsmRegClass {
        v0: reg = ["$2", "$v0"],
        v1: reg = ["$3", "$v1"],
        a0: reg = ["$4", "$a0"],
        a1: reg = ["$5", "$a1"],
        a2: reg = ["$6", "$a2"],
        a3: reg = ["$7", "$a3"],
        // FIXME: Reserve $t0, $t1 if in mips16 mode.
        t0: reg = ["$8", "$t0"],
        t1: reg = ["$9", "$t1"],
        t2: reg = ["$10", "$t2"],
        t3: reg = ["$11", "$t3"],
        t4: reg = ["$12", "$t4"],
        t5: reg = ["$13", "$t5"],
        t6: reg = ["$14", "$t6"],
        t7: reg = ["$15", "$t7"],
        s0: reg = ["$16", "$s0"],
        s1: reg = ["$17", "$s1"],
        s2: reg = ["$18", "$s2"],
        s3: reg = ["$19", "$s3"],
        s4: reg = ["$20", "$s4"],
        s5: reg = ["$21", "$s5"],
        s6: reg = ["$22", "$s6"],
        s7: reg = ["$23", "$s7"],
        t8: reg = ["$24", "$t8"],
        t9: reg = ["$25", "$t9"],
        f0: freg = ["$f0"],
        f1: freg = ["$f1"],
        f2: freg = ["$f2"],
        f3: freg = ["$f3"],
        f4: freg = ["$f4"],
        f5: freg = ["$f5"],
        f6: freg = ["$f6"],
        f7: freg = ["$f7"],
        f8: freg = ["$f8"],
        f9: freg = ["$f9"],
        f10: freg = ["$f10"],
        f11: freg = ["$f11"],
        f12: freg = ["$f12"],
        f13: freg = ["$f13"],
        f14: freg = ["$f14"],
        f15: freg = ["$f15"],
        f16: freg = ["$f16"],
        f17: freg = ["$f17"],
        f18: freg = ["$f18"],
        f19: freg = ["$f19"],
        f20: freg = ["$f20"],
        f21: freg = ["$f21"],
        f22: freg = ["$f22"],
        f23: freg = ["$f23"],
        f24: freg = ["$f24"],
        f25: freg = ["$f25"],
        f26: freg = ["$f26"],
        f27: freg = ["$f27"],
        f28: freg = ["$f28"],
        f29: freg = ["$f29"],
        f30: freg = ["$f30"],
        f31: freg = ["$f31"],
        #error = ["$0", "$zero"] =>
            "constant zero cannot be used as an operand for inline asm",
        #error = ["$1", "$at"] =>
            "reserved for assembler (Assembler Temp)",
        #error = ["$26", "$k0"] =>
            "OS-reserved register cannot be used as an operand for inline asm",
        #error = ["$27", "$k1"] =>
            "OS-reserved register cannot be used as an operand for inline asm",
        #error = ["$28", "$gp"] =>
            "the global pointer cannot be used as an operand for inline asm",
        #error = ["$29", "$sp"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["$30", "$s8", "$fp"] =>
            "the frame pointer cannot be used as an operand for inline asm",
        #error = ["$31", "$ra"] =>
            "the return address register cannot be used as an operand for inline asm",
    }
}

impl MipsInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        out.write_str(self.name())
    }
}
