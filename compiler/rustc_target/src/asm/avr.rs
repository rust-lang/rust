use std::fmt;

use rustc_data_structures::fx::FxIndexSet;
use rustc_span::{Symbol, sym};

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};
use crate::spec::{RelocModel, Target};

def_reg_class! {
    Avr AvrInlineAsmRegClass {
        reg,
        reg_upper,
        reg_pair,
        reg_iw,
        reg_ptr,
    }
}

impl AvrInlineAsmRegClass {
    pub fn valid_modifiers(self, _arch: InlineAsmArch) -> &'static [char] {
        match self {
            Self::reg_pair | Self::reg_iw | Self::reg_ptr => &['h', 'l'],
            _ => &[],
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
        _arch: InlineAsmArch,
    ) -> &'static [(InlineAsmType, Option<Symbol>)] {
        match self {
            Self::reg => types! { _: I8; },
            Self::reg_upper => types! { _: I8; },
            Self::reg_pair => types! { _: I16; },
            Self::reg_iw => types! { _: I16; },
            Self::reg_ptr => types! { _: I16; },
        }
    }
}

pub(crate) fn is_tiny(target_features: &FxIndexSet<Symbol>) -> bool {
    target_features.contains(&sym::tinyencoding)
}

fn not_tiny(
    _arch: InlineAsmArch,
    _reloc_model: RelocModel,
    target_features: &FxIndexSet<Symbol>,
    _target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    if is_tiny(target_features) {
        Err(
            "on AVRTiny, r[2-15] are unavailable, r16 (scratch register) and r17 (zero register) are reserved by LLVM",
        )
    } else {
        Ok(())
    }
}

def_regs! {
    Avr AvrInlineAsmReg AvrInlineAsmRegClass {
        r2: reg = ["r2"] % not_tiny,
        r3: reg = ["r3"] % not_tiny,
        r4: reg = ["r4"] % not_tiny,
        r5: reg = ["r5"] % not_tiny,
        r6: reg = ["r6"] % not_tiny,
        r7: reg = ["r7"] % not_tiny,
        r8: reg = ["r8"] % not_tiny,
        r9: reg = ["r9"] % not_tiny,
        r10: reg = ["r10"] % not_tiny,
        r11: reg = ["r11"] % not_tiny,
        r12: reg = ["r12"] % not_tiny,
        r13: reg = ["r13"] % not_tiny,
        r14: reg = ["r14"] % not_tiny,
        r15: reg = ["r15"] % not_tiny,
        r16: reg, reg_upper = ["r16"] % not_tiny,
        r17: reg, reg_upper = ["r17"] % not_tiny,
        r18: reg, reg_upper = ["r18"],
        r19: reg, reg_upper = ["r19"],
        r20: reg, reg_upper = ["r20"],
        r21: reg, reg_upper = ["r21"],
        r22: reg, reg_upper = ["r22"],
        r23: reg, reg_upper = ["r23"],
        r24: reg, reg_upper = ["r24"],
        r25: reg, reg_upper = ["r25"],
        r26: reg, reg_upper = ["r26", "XL"],
        r27: reg, reg_upper = ["r27", "XH"],
        r30: reg, reg_upper = ["r30", "ZL"],
        r31: reg, reg_upper = ["r31", "ZH"],

        r3r2: reg_pair = ["r3r2"] % not_tiny,
        r5r4: reg_pair = ["r5r4"] % not_tiny,
        r7r6: reg_pair = ["r7r6"] % not_tiny,
        r9r8: reg_pair = ["r9r8"] % not_tiny,
        r11r10: reg_pair = ["r11r10"] % not_tiny,
        r13r12: reg_pair = ["r13r12"] % not_tiny,
        r15r14: reg_pair = ["r15r14"] % not_tiny,
        r17r16: reg_pair = ["r17r16"] % not_tiny,
        r19r18: reg_pair = ["r19r18"],
        r21r20: reg_pair = ["r21r20"],
        r23r22: reg_pair = ["r23r22"],

        r25r24: reg_iw, reg_pair = ["r25r24"],

        X: reg_ptr, reg_iw, reg_pair = ["r27r26", "X"],
        Z: reg_ptr, reg_iw, reg_pair = ["r31r30", "Z"],

        #error = ["Y", "YL", "YH"] =>
            "the frame pointer cannot be used as an operand for inline asm",
        #error = ["SP", "SPL", "SPH"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["r0", "r1", "r1r0"] =>
            "LLVM reserves r0 (scratch register) and r1 (zero register)",
            // If this changes within LLVM, the compiler might use the registers
            // in the future. This must be reflected in the set of clobbered
            // registers, else the clobber ABI implementation is *unsound*, as
            // this generates invalid code (register is not marked as clobbered
            // but may change the register content).
    }
}

macro_rules! emit_pairs {
    (
        $self:ident $modifier:ident,
        $($pair:ident $name:literal $hi:literal $lo:literal,)*
    ) => {
        match ($self, $modifier) {
            $(
                (AvrInlineAsmReg::$pair, Some('h')) => $hi,
                (AvrInlineAsmReg::$pair, Some('l')) => $lo,
                (AvrInlineAsmReg::$pair, _) => $name,
            )*
            _ => $self.name(),
        }
    };
}

impl AvrInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        modifier: Option<char>,
    ) -> fmt::Result {
        let name = emit_pairs! {
            self modifier,
            Z "Z" "ZH" "ZL",
            X "X" "XH" "XL",
            r25r24 "r25:r24" "r25" "r24",
            r23r22 "r23:r22" "r23" "r22",
            r21r20 "r21:r20" "r21" "r20",
            r19r18 "r19:r18" "r19" "r18",
            r17r16 "r17:r16" "r17" "r16",
            r15r14 "r15:r14" "r15" "r14",
            r13r12 "r13:r12" "r13" "r12",
            r11r10 "r11:r10" "r11" "r10",
            r9r8 "r9:r8" "r9" "r8",
            r7r6 "r7:r6" "r7" "r6",
            r5r4 "r5:r4" "r5" "r4",
            r3r2 "r3:r2" "r3" "r2",
        };
        out.write_str(name)
    }

    pub fn overlapping_regs(self, mut cb: impl FnMut(AvrInlineAsmReg)) {
        cb(self);

        macro_rules! reg_conflicts {
            (
                $(
                    $pair:ident : $hi:ident $lo:ident,
                )*
            ) => {
                match self {
                    $(
                        Self::$pair => {
                            cb(Self::$hi);
                            cb(Self::$lo);
                        }
                        Self::$hi => {
                            cb(Self::$pair);
                        }
                        Self::$lo => {
                            cb(Self::$pair);
                        }
                    )*
                }
            };
        }

        reg_conflicts! {
            Z : r31 r30,
            X : r27 r26,
            r25r24 : r25 r24,
            r23r22 : r23 r22,
            r21r20 : r21 r20,
            r19r18 : r19 r18,
            r17r16 : r17 r16,
            r15r14 : r15 r14,
            r13r12 : r13 r12,
            r11r10 : r11 r10,
            r9r8 : r9 r8,
            r7r6 : r7 r6,
            r5r4 : r5 r4,
            r3r2 : r3 r2,
        }
    }
}
