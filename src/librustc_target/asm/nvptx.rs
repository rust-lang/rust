use super::{InlineAsmArch, InlineAsmType};
use rustc_macros::HashStable_Generic;
use std::fmt;

def_reg_class! {
    Nvptx NvptxInlineAsmRegClass {
        reg16,
        reg32,
        reg64,
        freg32,
        freg64,
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
    ) -> &'static [(InlineAsmType, Option<&'static str>)] {
        match self {
            Self::reg16 => types! { _: I8, I16; },
            Self::reg32 => types! { _: I8, I16, I32; },
            Self::reg64 => types! { _: I8, I16, I32, I64; },
            Self::freg32 => types! { _: F32; },
            Self::freg64 => types! { _: F32, F64; },
        }
    }
}

def_regs! {
    Nvptx NvptxInlineAsmReg NvptxInlineAsmRegClass {
        // We have to define a register, otherwise we get warnings/errors about unused imports and
        // unreachable code. Do what clang does and define r0.
        r0: reg32 = ["r0"],
        #error = ["tid", "tid.x", "tid.y", "tid.z"] => "tid not supported for inline asm",
        #error = ["ntid", "ntid.x", "ntid.y", "ntid.z"] => "ntid not supported for inline asm",
        #error = ["laneid"] => "laneid not supported for inline asm",
        #error = ["warpid"] => "warpid not supported for inline asm",
        #error = ["nwarpid"] => "nwarpid not supported for inline asm",
        #error = ["ctaid", "ctaid.x", "ctaid.y", "ctaid.z"] => "ctaid not supported for inline asm",
        #error = ["nctaid", "nctaid.x", "nctaid.y", "nctaid.z"] => "nctaid not supported for inline asm",
        #error = ["smid"] => "smid not supported for inline asm",
        #error = ["nsmid"] => "nsmid not supported for inline asm",
        #error = ["gridid"] => "gridid not supported for inline asm",
        #error = ["lanemask_eq"] => "lanemask_eq not supported for inline asm",
        #error = ["lanemask_le"] => "lanemask_le not supported for inline asm",
        #error = ["lanemask_lt"] => "lanemask_lt not supported for inline asm",
        #error = ["lanemask_ge"] => "lanemask_ge not supported for inline asm",
        #error = ["lanemask_gt"] => "lanemask_gt not supported for inline asm",
        #error = ["clock", "clock_hi"] => "clock not supported for inline asm",
        #error = ["clock64"] => "clock64 not supported for inline asm",
        #error = ["pm0", "pm1", "pm2", "pm3", "pm4", "pm5", "pm6", "pm7"] => "pm not supported for inline asm",
        #error = ["pm0_64", "pm1_64", "pm2_64", "pm3_64", "pm4_64", "pm5_64", "pm6_64", "pm7_64"] => "pm_64 not supported for inline asm",
        #error = ["envreg0", "envreg1", "envreg2", "envreg3", "envreg4", "envreg5", "envreg6", "envreg7", "envreg8", "envreg9", "envreg10", "envreg11", "envreg12", "envreg13", "envreg14", "envreg15", "envreg16", "envreg17", "envreg18", "envreg19", "envreg20", "envreg21", "envreg22", "envreg23", "envreg24", "envreg25", "envreg26", "envreg27", "envreg28", "envreg29", "envreg30", "envreg31"] => "envreg not supported for inline asm",
        #error = ["globaltimer", "globaltimer_lo", "globaltimer_hi"] => "globaltimer not supported for inline asm",
        #error = ["total_mem_size"] => "total_mem_size not supported for inline asm",
        #error = ["dynamic_mem_size"] => "dynamic_mem_size not supported for inline asm",
    }
}

impl NvptxInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        out.write_str(self.name())
    }
}
