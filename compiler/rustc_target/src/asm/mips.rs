use std::fmt;

use rustc_span::Symbol;

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};

def_reg_class! {
    Mips MipsInlineAsmRegClass {
        reg,
        freg,
        wreg,
    }
}

impl MipsInlineAsmRegClass {
    pub fn valid_modifiers(self, _arch: super::InlineAsmArch) -> &'static [char] {
        match self {
            Self::reg => &[],
            Self::freg => &['w'],
            // LLVM doesn't currently support displaying vector registers holding vector types as
            // float registers.
            Self::wreg => &[],
        }
    }

    pub fn suggest_class(self, _arch: InlineAsmArch, _ty: InlineAsmType) -> Option<Self> {
        None
    }

    pub fn suggest_modifier(
        self,
        _arch: InlineAsmArch,
        _ty: InlineAsmType,
    ) -> Option<ModifierInfo> {
        None
    }

    pub fn default_modifier(self, _arch: InlineAsmArch) -> Option<ModifierInfo> {
        None
    }

    pub fn supported_types(
        self,
        arch: InlineAsmArch,
    ) -> &'static [(InlineAsmType, Option<Symbol>)] {
        match (self, arch) {
            (Self::reg, InlineAsmArch::Mips64) => types! { _: I8, I16, I32, I64, F16, F32, F64; },
            (Self::reg, _) => types! { _: I8, I16, I32, F16, F32; },
            (Self::freg, _) => types! { _: F16, F32, F64; },
            (Self::wreg, _) => {
                types! { msa: F16, F32, F64, VecI8(16), VecI16(8), VecI32(4), VecI64(2), VecF16(8), VecF32(4), VecF64(2); }
            }
        }
    }
}

// The reserved registers are somewhat taken from
// <https://github.com/llvm/llvm-project/blob/deb8f8bcf31540c657716ea5242183b0792702a1/llvm/lib/Target/Mips/MipsRegisterInfo.cpp#L150>.
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
        w0: wreg = ["$w0"],
        w1: wreg = ["$w1"],
        w2: wreg = ["$w2"],
        w3: wreg = ["$w3"],
        w4: wreg = ["$w4"],
        w5: wreg = ["$w5"],
        w6: wreg = ["$w6"],
        w7: wreg = ["$w7"],
        w8: wreg = ["$w8"],
        w9: wreg = ["$w9"],
        w10: wreg = ["$w10"],
        w11: wreg = ["$w11"],
        w12: wreg = ["$w12"],
        w13: wreg = ["$w13"],
        w14: wreg = ["$w14"],
        w15: wreg = ["$w15"],
        w16: wreg = ["$w16"],
        w17: wreg = ["$w17"],
        w18: wreg = ["$w18"],
        w19: wreg = ["$w19"],
        w20: wreg = ["$w20"],
        w21: wreg = ["$w21"],
        w22: wreg = ["$w22"],
        w23: wreg = ["$w23"],
        w24: wreg = ["$w24"],
        w25: wreg = ["$w25"],
        w26: wreg = ["$w26"],
        w27: wreg = ["$w27"],
        w28: wreg = ["$w28"],
        w29: wreg = ["$w29"],
        w30: wreg = ["$w30"],
        w31: wreg = ["$w31"],
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

    pub fn overlapping_regs(self, mut cb: impl FnMut(MipsInlineAsmReg)) {
        cb(self);

        macro_rules! reg_conflicts {
            (
                $(
                    $full:ident : $($field:ident)*
                ),*;
            ) => {
                match self {
                    $(
                        Self::$full => {
                            $(cb(Self::$field);)*
                        }
                        $(Self::$field)|* => cb(Self::$full),
                    )*
                    _ => {}
                }
            };
        }

        // Float registers overlap the first half of vector registers.
        reg_conflicts! {
            w0: f0,
            w1: f1,
            w2: f2,
            w3: f3,
            w4: f4,
            w5: f5,
            w6: f6,
            w7: f7,
            w8: f8,
            w9: f9,
            w10: f10,
            w11: f11,
            w12: f12,
            w13: f13,
            w14: f14,
            w15: f15,
            w16: f16,
            w17: f17,
            w18: f18,
            w19: f19,
            w20: f20,
            w21: f21,
            w22: f22,
            w23: f23,
            w24: f24,
            w25: f25,
            w26: f26,
            w27: f27,
            w28: f28,
            w29: f29,
            w30: f30,
            w31: f31;
        }
    }
}
