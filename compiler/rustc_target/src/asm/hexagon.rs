use std::fmt;

use rustc_span::Symbol;

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};

def_reg_class! {
    Hexagon HexagonInlineAsmRegClass {
        reg,
        reg_pair,
        preg,
        vreg,
        qreg,
    }
}

impl HexagonInlineAsmRegClass {
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
            Self::reg => types! { _: I8, I16, I32, F32; },
            Self::reg_pair => types! { _: I64, F64; },
            Self::preg => &[],
            Self::vreg => types! {
                hvx_length64b: VecI32(16);
                hvx_length128b: VecI32(32);
            },
            Self::qreg => &[],
        }
    }
}

def_regs! {
    Hexagon HexagonInlineAsmReg HexagonInlineAsmRegClass {
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
        r10: reg = ["r10"],
        r11: reg = ["r11"],
        r12: reg = ["r12"],
        r13: reg = ["r13"],
        r14: reg = ["r14"],
        r15: reg = ["r15"],
        r16: reg = ["r16"],
        r17: reg = ["r17"],
        r18: reg = ["r18"],
        r20: reg = ["r20"],
        r21: reg = ["r21"],
        r22: reg = ["r22"],
        r23: reg = ["r23"],
        r24: reg = ["r24"],
        r25: reg = ["r25"],
        r26: reg = ["r26"],
        r27: reg = ["r27"],
        r28: reg = ["r28"],
        r1_0: reg_pair = ["r1:0"],
        r3_2: reg_pair = ["r3:2"],
        r5_4: reg_pair = ["r5:4"],
        r7_6: reg_pair = ["r7:6"],
        r9_8: reg_pair = ["r9:8"],
        r11_10: reg_pair = ["r11:10"],
        r13_12: reg_pair = ["r13:12"],
        r15_14: reg_pair = ["r15:14"],
        r17_16: reg_pair = ["r17:16"],
        r21_20: reg_pair = ["r21:20"],
        r23_22: reg_pair = ["r23:22"],
        r25_24: reg_pair = ["r25:24"],
        r27_26: reg_pair = ["r27:26"],
        p0: preg = ["p0"],
        p1: preg = ["p1"],
        p2: preg = ["p2"],
        p3: preg = ["p3"],
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
        q0: qreg = ["q0"],
        q1: qreg = ["q1"],
        q2: qreg = ["q2"],
        q3: qreg = ["q3"],
        #error = ["r19"] =>
            "r19 is used internally by LLVM and cannot be used as an operand for inline asm",
        #error = ["r19:18"] =>
            "r19 is used internally by LLVM and cannot be used as an operand for inline asm",
        #error = ["r29", "sp"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["r29:28"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["r30", "fr"] =>
            "the frame register cannot be used as an operand for inline asm",
        #error = ["r31", "lr"] =>
            "the link register cannot be used as an operand for inline asm",
        #error = ["r31:30"] =>
            "the frame register and link register cannot be used as an operand for inline asm",
    }
}

impl HexagonInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        out.write_str(self.name())
    }

    pub fn overlapping_regs(self, mut cb: impl FnMut(HexagonInlineAsmReg)) {
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
                    _ => {}
                }
            };
        }

        reg_conflicts! {
            r1_0 : r1 r0,
            r3_2 : r3 r2,
            r5_4 : r5 r4,
            r7_6 : r7 r6,
            r9_8 : r9 r8,
            r11_10 : r11 r10,
            r13_12 : r13 r12,
            r15_14 : r15 r14,
            r17_16 : r17 r16,
            r21_20 : r21 r20,
            r23_22 : r23 r22,
            r25_24 : r25 r24,
            r27_26 : r27 r26,
        }
    }
}
