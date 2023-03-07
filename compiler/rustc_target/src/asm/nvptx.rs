use super::{InlineAsmArch, InlineAsmType};
use rustc_macros::HashStable_Generic;
use rustc_span::Symbol;

def_reg_class! {
    Nvptx NvptxInlineAsmRegClass {
        reg16,
        reg32,
        reg64,
    }
}

impl NvptxInlineAsmRegClass {
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
        _arch: InlineAsmArch,
    ) -> &'static [(InlineAsmType, Option<Symbol>)] {
        match self {
            Self::reg16 => types! { _: I8, I16; },
            Self::reg32 => types! { _: I8, I16, I32, F32; },
            Self::reg64 => types! { _: I8, I16, I32, F32, I64, F64; },
        }
    }
}

def_regs! {
    // Registers in PTX are declared in the assembly.
    // There are no predefined registers that one can use.
    Nvptx NvptxInlineAsmReg NvptxInlineAsmRegClass {}
}
