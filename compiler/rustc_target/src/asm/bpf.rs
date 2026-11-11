use std::fmt;

use rustc_span::Symbol;

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};

def_reg_class! {
    Bpf BpfInlineAsmRegClass {
        reg,
        wreg,
    }
}

impl BpfInlineAsmRegClass {
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
            Self::reg => types! { _: I8, I16, I32, I64; },
            Self::wreg => types! { alu32: I8, I16, I32; },
        }
    }
}

def_regs! {
    Bpf BpfInlineAsmReg BpfInlineAsmRegClass {
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
        w0: wreg = ["w0"],
        w1: wreg = ["w1"],
        w2: wreg = ["w2"],
        w3: wreg = ["w3"],
        w4: wreg = ["w4"],
        w5: wreg = ["w5"],
        w6: wreg = ["w6"],
        w7: wreg = ["w7"],
        w8: wreg = ["w8"],
        w9: wreg = ["w9"],

        #error = ["r10", "w10"] =>
            "the stack pointer cannot be used as an operand for inline asm",
    }
}

impl BpfInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        out.write_str(self.name())
    }

    pub fn overlapping_regs(self, mut cb: impl FnMut(BpfInlineAsmReg)) {
        cb(self);

        macro_rules! reg_conflicts {
            (
                $(
                    $r:ident : $w:ident
                ),*
            ) => {
                match self {
                    $(
                        Self::$r => {
                            cb(Self::$w);
                        }
                        Self::$w => {
                            cb(Self::$r);
                        }
                    )*
                }
            };
        }

        reg_conflicts! {
            r0 : w0,
            r1 : w1,
            r2 : w2,
            r3 : w3,
            r4 : w4,
            r5 : w5,
            r6 : w6,
            r7 : w7,
            r8 : w8,
            r9 : w9
        }
    }
}
