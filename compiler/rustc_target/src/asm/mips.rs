use super::{InlineAsmArch, InlineAsmType};
use rustc_macros::HashStable_Generic;
use rustc_span::Symbol;
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
        arch: InlineAsmArch,
    ) -> &'static [(InlineAsmType, Option<Symbol>)] {
        match (self, arch) {
            (Self::reg, InlineAsmArch::Mips64) => types! { _: I8, I16, I32, I64, F32, F64; },
            (Self::reg, _) => types! { _: I8, I16, I32, F32; },
            (Self::freg, _) => types! { _: F32, F64; },
        }
    }
}

// The reserved registers are somewhat taken from <https://git.io/JUR1k#L150>.
def_regs! {
    Mips MipsInlineAsmReg MipsInlineAsmRegClass {
        r2: reg = ["$2"],
        r3: reg = ["$3"],
        r4: reg = ["$4"],
        r5: reg = ["$5"],
        r6: reg = ["$6"],
        r7: reg = ["$7"],
        // FIXME: Reserve $t0, $t1 if in mips16 mode.
        r8: reg = ["$8"],
        r9: reg = ["$9"],
        r10: reg = ["$10"],
        r11: reg = ["$11"],
        r12: reg = ["$12"],
        r13: reg = ["$13"],
        r14: reg = ["$14"],
        r15: reg = ["$15"],
        r16: reg = ["$16"],
        r17: reg = ["$17"],
        r18: reg = ["$18"],
        r19: reg = ["$19"],
        r20: reg = ["$20"],
        r21: reg = ["$21"],
        r22: reg = ["$22"],
        r23: reg = ["$23"],
        r24: reg = ["$24"],
        r25: reg = ["$25"],
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
        #error = ["$0"] =>
            "constant zero cannot be used as an operand for inline asm",
        #error = ["$1"] =>
            "reserved for assembler (Assembler Temp)",
        #error = ["$26"] =>
            "OS-reserved register cannot be used as an operand for inline asm",
        #error = ["$27"] =>
            "OS-reserved register cannot be used as an operand for inline asm",
        #error = ["$28"] =>
            "the global pointer cannot be used as an operand for inline asm",
        #error = ["$29"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["$30"] =>
            "the frame pointer cannot be used as an operand for inline asm",
        #error = ["$31"] =>
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
