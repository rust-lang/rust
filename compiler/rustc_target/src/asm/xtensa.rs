use std::fmt;

use rustc_data_structures::fx::FxIndexSet;
use rustc_span::{Symbol, sym};

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};
use crate::spec::{FramePointer, RelocModel, Target};

def_reg_class! {
    Xtensa XtensaInlineAsmRegClass {
        reg,
        freg,
    }
}

impl XtensaInlineAsmRegClass {
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
            Self::freg => types! { fp: F32; },
        }
    }
}

fn has_fp(
    _arch: InlineAsmArch,
    _reloc_model: RelocModel,
    target_features: &FxIndexSet<Symbol>,
    _target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    if target_features.contains(&sym::fp) {
        Ok(())
    } else {
        Err("target does not support floating point registers")
    }
}

fn is_frame_pointer(
    _arch: InlineAsmArch,
    _reloc_model: RelocModel,
    _target_features: &FxIndexSet<Symbol>,
    target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    // The Xtensa arch doesn't require, nor use, a dedicated frame pointer register
    // therefore if it is not force enabled, we can assume it won't be generated.
    // If frame pointers are enabled, we cannot use the register as a general purpose one.
    if target.options.frame_pointer != FramePointer::MayOmit {
        Err("frame pointer register cannot be used when frame pointers are enabled")
    } else {
        Ok(())
    }
}

def_regs! {
    Xtensa XtensaInlineAsmReg XtensaInlineAsmRegClass {
        a2: reg = ["a2"],
        a3: reg = ["a3"],
        a4: reg = ["a4"],
        a5: reg = ["a5"],
        a6: reg = ["a6"],
        a7: reg = ["a7"],
        a8: reg = ["a8"],
        a9: reg = ["a9"],
        a10: reg = ["a10"],
        a11: reg = ["a11"],
        a12: reg = ["a12"],
        a13: reg = ["a13"],
        a14: reg = ["a14"],
        a15: reg = ["a15"] % is_frame_pointer,
        sar: reg = ["sar"],
        f0: freg = ["f0"] % has_fp,
        f1: freg = ["f1"] % has_fp,
        f2: freg = ["f2"] % has_fp,
        f3: freg = ["f3"] % has_fp,
        f4: freg = ["f4"] % has_fp,
        f5: freg = ["f5"] % has_fp,
        f6: freg = ["f6"] % has_fp,
        f7: freg = ["f7"] % has_fp,
        f8: freg = ["f8"] % has_fp,
        f9: freg = ["f9"] % has_fp,
        f10: freg = ["f10"] % has_fp,
        f11: freg = ["f11"] % has_fp,
        f12: freg = ["f12"] % has_fp,
        f13: freg = ["f13"] % has_fp,
        f14: freg = ["f14"] % has_fp,
        f15: freg = ["f15"] % has_fp,

        #error = ["a0"] => "a0 is used internally by LLVM and cannot be used as an operand for inline asm",
        #error = ["sp", "a1"] => "sp is used internally by LLVM and cannot be used as an operand for inline asm",
    }
}

impl XtensaInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        out.write_str(self.name())
    }
}
