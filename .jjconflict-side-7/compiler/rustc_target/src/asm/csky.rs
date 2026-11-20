use std::fmt;

use rustc_span::Symbol;

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};

def_reg_class! {
    CSKY CSKYInlineAsmRegClass {
        reg,
        freg,
    }
}

impl CSKYInlineAsmRegClass {
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
    ) -> Option<ModifierInfo> {
        None
    }

    pub fn default_modifier(self, _arch: InlineAsmArch) -> Option<ModifierInfo> {
        None
    }

    pub fn supported_types(
        self,
        _arch: InlineAsmArch,
    ) -> &'static [(InlineAsmType, Option<Symbol>)] {
        match self {
            Self::reg => types! { _: I8, I16, I32; },
            Self::freg => types! { _: F32; },
        }
    }
}

// The reserved registers are taken from <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/CSKY/CSKYRegisterInfo.cpp#79>
def_regs! {
    CSKY CSKYInlineAsmReg CSKYInlineAsmRegClass {
        r0: reg = ["r0","a0"],
        r1: reg = ["r1","a1"],
        r2: reg = ["r2","a2"],
        r3: reg = ["r3","a3"],
        r4: reg = ["r4","l0"],
        r5: reg = ["r5","l1"],
        r6: reg = ["r6","l2"],
        r9: reg = ["r9","l5"],// feature e2
        r10: reg = ["r10","l6"],// feature e2
        r11: reg = ["r11","l7"],// feature e2
        r12: reg = ["r12","t0"],// feature e2
        r13: reg = ["r13","t1"],// feature e2
        r16: reg = ["r16","l8"],// feature high-register
        r17: reg = ["r17","l9"],// feature high-register
        r18: reg = ["r18","t2"],// feature high-register
        r19: reg = ["r19","t3"],// feature high-register
        r20: reg = ["r20","t4"],// feature high-register
        r21: reg = ["r21","t5"],// feature high-register
        r22: reg = ["r22","t6"],// feature high-register
        r23: reg = ["r23","t7"],// feature high-register
        r24: reg = ["r24","t8"],// feature high-register
        r25: reg = ["r25","t9"],// feature high-register
        f0: freg = ["fr0","vr0"],
        f1: freg = ["fr1","vr1"],
        f2: freg = ["fr2","vr2"],
        f3: freg = ["fr3","vr3"],
        f4: freg = ["fr4","vr4"],
        f5: freg = ["fr5","vr5"],
        f6: freg = ["fr6","vr6"],
        f7: freg = ["fr7","vr7"],
        f8: freg = ["fr8","vr8"],
        f9: freg = ["fr9","vr9"],
        f10: freg = ["fr10","vr10"],
        f11: freg = ["fr11","vr11"],
        f12: freg = ["fr12","vr12"],
        f13: freg = ["fr13","vr13"],
        f14: freg = ["fr14","vr14"],
        f15: freg = ["fr15","vr15"],
        f16: freg = ["fr16","vr16"],
        f17: freg = ["fr17","vr17"],
        f18: freg = ["fr18","vr18"],
        f19: freg = ["fr19","vr19"],
        f20: freg = ["fr20","vr20"],
        f21: freg = ["fr21","vr21"],
        f22: freg = ["fr22","vr22"],
        f23: freg = ["fr23","vr23"],
        f24: freg = ["fr24","vr24"],
        f25: freg = ["fr25","vr25"],
        f26: freg = ["fr26","vr26"],
        f27: freg = ["fr27","vr27"],
        f28: freg = ["fr28","vr28"],
        f29: freg = ["fr29","vr29"],
        f30: freg = ["fr30","vr30"],
        f31: freg = ["fr31","vr31"],
        #error = ["r7", "l3"] =>
            "the base pointer cannot be used as an operand for inline asm",
        #error = ["r8","l4"] =>
            "the frame pointer cannot be used as an operand for inline asm",
        #error = ["r14","sp"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["r15","lr"] =>
            "the link register cannot be used as an operand for inline asm",
        #error = ["r31","tls"] =>
            "reserver for tls",
        #error = ["r28", "gb", "rgb", "rdb"] =>
            "the global pointer cannot be used as an operand for inline asm",
        #error = ["r26","r27","r29","tb", "rtb", "r30","svbr"] =>
            "reserved by the ABI",
    }
}

impl CSKYInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        out.write_str(self.name())
    }
}
