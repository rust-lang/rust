use std::fmt;

use rustc_data_structures::fx::FxIndexSet;
use rustc_span::{Symbol, kw, sym};

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};
use crate::spec::{FramePointer, RelocModel, Target};

def_reg_class! {
    Xtensa XtensaInlineAsmRegClass {
        reg,
        freg,
        sreg,
        breg,
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
            Self::sreg | Self::breg => &[],
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

fn has_bool(
    _arch: InlineAsmArch,
    _reloc_model: RelocModel,
    target_features: &FxIndexSet<Symbol>,
    _target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    if target_features.contains(&sym::bool) {
        Ok(())
    } else {
        Err("target does not support boolean registers")
    }
}

fn has_loop(
    _arch: InlineAsmArch,
    _reloc_model: RelocModel,
    target_features: &FxIndexSet<Symbol>,
    _target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    if target_features.contains(&kw::Loop) {
        Ok(())
    } else {
        Err("target does not support loop registers")
    }
}

fn has_mac16(
    _arch: InlineAsmArch,
    _reloc_model: RelocModel,
    target_features: &FxIndexSet<Symbol>,
    _target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    if target_features.contains(&sym::mac16) {
        Ok(())
    } else {
        Err("target does not support MAC16 registers")
    }
}

fn has_s32c1i(
    _arch: InlineAsmArch,
    _reloc_model: RelocModel,
    target_features: &FxIndexSet<Symbol>,
    _target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    if target_features.contains(&sym::s32c1i) {
        Ok(())
    } else {
        Err("target does not support the s32c1i instruction")
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
        // Clobber-only special registers.
        // Per the Xtensa ISA reference (Section 8.1.6), all non-privileged special
        // registers except LITBASE are caller-saved. These must be available as
        // clobbers for `clobber_abi`.
        sar: sreg = ["sar"],
        scompare1: sreg = ["scompare1"] % has_s32c1i,
        lbeg: sreg = ["lbeg"] % has_loop,
        lend: sreg = ["lend"] % has_loop,
        lcount: sreg = ["lcount"] % has_loop,
        acclo: sreg = ["acclo"] % has_mac16,
        acchi: sreg = ["acchi"] % has_mac16,
        m0: sreg = ["m0"] % has_mac16,
        m1: sreg = ["m1"] % has_mac16,
        m2: sreg = ["m2"] % has_mac16,
        m3: sreg = ["m3"] % has_mac16,
        // Clobber-only boolean registers.
        b0: breg = ["b0"] % has_bool,
        b1: breg = ["b1"] % has_bool,
        b2: breg = ["b2"] % has_bool,
        b3: breg = ["b3"] % has_bool,
        b4: breg = ["b4"] % has_bool,
        b5: breg = ["b5"] % has_bool,
        b6: breg = ["b6"] % has_bool,
        b7: breg = ["b7"] % has_bool,
        b8: breg = ["b8"] % has_bool,
        b9: breg = ["b9"] % has_bool,
        b10: breg = ["b10"] % has_bool,
        b11: breg = ["b11"] % has_bool,
        b12: breg = ["b12"] % has_bool,
        b13: breg = ["b13"] % has_bool,
        b14: breg = ["b14"] % has_bool,
        b15: breg = ["b15"] % has_bool,

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
