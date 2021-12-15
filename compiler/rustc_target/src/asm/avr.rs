use super::{InlineAsmArch, InlineAsmType};
use rustc_macros::HashStable_Generic;
use std::fmt;

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
            Self::reg => types! { _: I8; },
            Self::reg_upper => types! { _: I8; },
            Self::reg_pair => types! { _: I16; },
            Self::reg_iw => types! { _: I16; },
            Self::reg_ptr => types! { _: I16; },
        }
    }
}

def_regs! {
    Avr AvrInlineAsmReg AvrInlineAsmRegClass {
        r2: reg = ["r2"],
        r3: reg = ["r3"],
        r4: reg = ["r4"],
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
        r16: reg, reg_upper = ["r16"],
        r17: reg, reg_upper = ["r17"],
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

        r3r2: reg_pair = ["r3r2"],
        r5r4: reg_pair = ["r5r4"],
        r7r6: reg_pair = ["r7r6"],
        r9r8: reg_pair = ["r9r8"],
        r11r10: reg_pair = ["r11r10"],
        r13r12: reg_pair = ["r13r12"],
        r15r14: reg_pair = ["r15r14"],
        r17r16: reg_pair = ["r17r16"],
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
            "r0 and r1 are not available due to an issue in LLVM",
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
