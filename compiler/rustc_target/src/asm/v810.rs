use super::{InlineAsmArch, InlineAsmType};
use rustc_span::Symbol;
use std::fmt;

def_reg_class! {
    V810 V810InlineAsmRegClass {
        reg,
        sreg,
    }
}

impl V810InlineAsmRegClass {
    pub fn valid_modifiers(self, _arch: InlineAsmArch) -> &'static [char] {
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
        _arch: InlineAsmArch
    ) -> &'static [(InlineAsmType, Option<Symbol>)] {
        match self {
            Self::reg => types! { _: I8, I16, I32, F32; },
            Self::sreg => types! { _: I32; }
        }
    }
}

def_regs! {
    V810 V810InlineAsmReg V810InlineAsmRegClass {
        r0: reg = ["r0"],
        r1: reg = ["r1"],
        r2: reg = ["r2", "hp", "fp"],
        r3: reg = ["r3", "sp"],
        r4: reg = ["r4", "tp"],
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
        r16: reg = ["r16"],
        r17: reg = ["r17"],
        r18: reg = ["r18"],
        r19: reg = ["r19"],
        r20: reg = ["r20"],
        r21: reg = ["r21"],
        r22: reg = ["r22"],
        r23: reg = ["r23"],
        r24: reg = ["r24"],
        r25: reg = ["r25"],
        r26: reg = ["r26"],
        r27: reg = ["r27"],
        r28: reg = ["r28"],
        r29: reg = ["r29"],
        r30: reg = ["r30"],
        r31: reg = ["r31", "lp"],

        sr0: sreg = ["sr0", "eipc"],
        sr1: sreg = ["sr1", "eipsw"],
        sr2: sreg = ["sr2", "fepc"],
        sr3: sreg = ["sr3", "fepsw"],
        sr4: sreg = ["sr4", "ecr"],
        sr5: sreg = ["sr5", "psw"],
        sr6: sreg = ["sr6", "pir"],
        sr7: sreg = ["sr7", "tkcw"],
        sr24: sreg = ["sr24", "chcw"],
        sr25: sreg = ["sr25", "adtre"],
        sr29: sreg = ["sr29"],
        sr30: sreg = ["sr30"],
        sr31: sreg = ["sr31"],
    }
}

impl V810InlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        out.write_str(self.name())
    }

    pub fn overlapping_regs(self, mut _cb: impl FnMut(V810InlineAsmReg)) {}
}