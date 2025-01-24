use std::fmt;

use rustc_span::Symbol;

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};

def_reg_class! {
    S390x S390xInlineAsmRegClass {
        reg,
        reg_addr,
        freg,
        vreg,
        areg,
    }
}

impl S390xInlineAsmRegClass {
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
        allow_experimental_reg: bool,
    ) -> &'static [(InlineAsmType, Option<Symbol>)] {
        match self {
            Self::reg | Self::reg_addr => types! { _: I8, I16, I32, I64; },
            Self::freg => types! { _: F32, F64; },
            Self::vreg => {
                if allow_experimental_reg {
                    // non-clobber-only vector register support is unstable.
                    types! {
                        vector: I32, F32, I64, F64, I128, F128,
                            VecI8(16), VecI16(8), VecI32(4), VecI64(2), VecF32(4), VecF64(2);
                    }
                } else {
                    &[]
                }
            }
            Self::areg => &[],
        }
    }
}

def_regs! {
    S390x S390xInlineAsmReg S390xInlineAsmRegClass {
        r0: reg = ["r0"],
        r1: reg, reg_addr = ["r1"],
        r2: reg, reg_addr = ["r2"],
        r3: reg, reg_addr = ["r3"],
        r4: reg, reg_addr = ["r4"],
        r5: reg, reg_addr = ["r5"],
        r6: reg, reg_addr = ["r6"],
        r7: reg, reg_addr = ["r7"],
        r8: reg, reg_addr = ["r8"],
        r9: reg, reg_addr = ["r9"],
        r10: reg, reg_addr = ["r10"],
        r12: reg, reg_addr = ["r12"],
        r13: reg, reg_addr = ["r13"],
        r14: reg, reg_addr = ["r14"],
        f0: freg = ["f0"],
        f1: freg = ["f1"],
        f2: freg = ["f2"],
        f3: freg = ["f3"],
        f4: freg = ["f4"],
        f5: freg = ["f5"],
        f6: freg = ["f6"],
        f7: freg = ["f7"],
        f8: freg = ["f8"],
        f9: freg = ["f9"],
        f10: freg = ["f10"],
        f11: freg = ["f11"],
        f12: freg = ["f12"],
        f13: freg = ["f13"],
        f14: freg = ["f14"],
        f15: freg = ["f15"],
        v0: vreg = ["v0"],
        v1: vreg = ["v1"],
        v2: vreg = ["v2"],
        v3: vreg = ["v3"],
        v4: vreg = ["v4"],
        v5: vreg = ["v5"],
        v6: vreg = ["v6"],
        v7: vreg = ["v7"],
        v8: vreg = ["v8"],
        v9: vreg = ["v9"],
        v10: vreg = ["v10"],
        v11: vreg = ["v11"],
        v12: vreg = ["v12"],
        v13: vreg = ["v13"],
        v14: vreg = ["v14"],
        v15: vreg = ["v15"],
        v16: vreg = ["v16"],
        v17: vreg = ["v17"],
        v18: vreg = ["v18"],
        v19: vreg = ["v19"],
        v20: vreg = ["v20"],
        v21: vreg = ["v21"],
        v22: vreg = ["v22"],
        v23: vreg = ["v23"],
        v24: vreg = ["v24"],
        v25: vreg = ["v25"],
        v26: vreg = ["v26"],
        v27: vreg = ["v27"],
        v28: vreg = ["v28"],
        v29: vreg = ["v29"],
        v30: vreg = ["v30"],
        v31: vreg = ["v31"],
        a2: areg = ["a2"],
        a3: areg = ["a3"],
        a4: areg = ["a4"],
        a5: areg = ["a5"],
        a6: areg = ["a6"],
        a7: areg = ["a7"],
        a8: areg = ["a8"],
        a9: areg = ["a9"],
        a10: areg = ["a10"],
        a11: areg = ["a11"],
        a12: areg = ["a12"],
        a13: areg = ["a13"],
        a14: areg = ["a14"],
        a15: areg = ["a15"],
        #error = ["r11"] =>
            "The frame pointer cannot be used as an operand for inline asm",
        #error = ["r15"] =>
            "The stack pointer cannot be used as an operand for inline asm",
        #error = [
            "c0", "c1", "c2", "c3",
            "c4", "c5", "c6", "c7",
            "c8", "c9", "c10", "c11",
            "c12", "c13", "c14", "c15"
        ] =>
            "control registers are reserved by the kernel and cannot be used as operands for inline asm",
        #error = ["a0", "a1"] =>
            "a0 and a1 are reserved for system use and cannot be used as operands for inline asm",
    }
}

impl S390xInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        write!(out, "%{}", self.name())
    }

    pub fn overlapping_regs(self, mut cb: impl FnMut(S390xInlineAsmReg)) {
        macro_rules! reg_conflicts {
            (
                $(
                    $full:ident : $($field:ident)*
                ),*;
            ) => {
                match self {
                    $(
                        Self::$full => {
                            cb(Self::$full);
                            $(cb(Self::$field);)*
                        }
                        $(Self::$field)|* => {
                            cb(Self::$full);
                            cb(self);
                        }
                    )*
                    r => cb(r),
                }
            };
        }

        // The left halves of v0-v15 are aliased to f0-f15.
        reg_conflicts! {
            v0 : f0,
            v1 : f1,
            v2 : f2,
            v3 : f3,
            v4 : f4,
            v5 : f5,
            v6 : f6,
            v7 : f7,
            v8 : f8,
            v9 : f9,
            v10 : f10,
            v11 : f11,
            v12 : f12,
            v13 : f13,
            v14 : f14,
            v15 : f15;
        }
    }
}
