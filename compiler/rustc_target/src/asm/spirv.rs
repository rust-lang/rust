use super::{InlineAsmArch, InlineAsmType};
use rustc_macros::HashStable_Generic;
use rustc_span::Symbol;

def_reg_class! {
    SpirV SpirVInlineAsmRegClass {
        reg,
    }
}

impl SpirVInlineAsmRegClass {
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
            Self::reg => {
                types! { _: I8, I16, I32, I64, F32, F64; }
            }
        }
    }
}

def_regs! {
    // SPIR-V is SSA-based, it does not have registers.
    SpirV SpirVInlineAsmReg SpirVInlineAsmRegClass {}
}
