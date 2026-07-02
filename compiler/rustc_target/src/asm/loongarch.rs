use std::fmt;

use rustc_span::Symbol;

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};

def_reg_class! {
    LoongArch LoongArchInlineAsmRegClass {
        reg,
        freg,
        vreg,
        xreg,
    }
}

impl LoongArchInlineAsmRegClass {
    pub fn valid_modifiers(self, _arch: super::InlineAsmArch) -> &'static [char] {
        match self {
            Self::freg => &['w', 'u'],
            Self::vreg => &['u'],
            Self::xreg => &['w'],
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
        arch: InlineAsmArch,
        allow_experimental_reg: bool,
    ) -> &'static [(InlineAsmType, Option<Symbol>)] {
        match (self, arch) {
            (Self::reg, InlineAsmArch::LoongArch64) => {
                types! { _: I8, I16, I32, I64, F16, F32, F64; }
            }
            (Self::reg, InlineAsmArch::LoongArch32) => types! { _: I8, I16, I32, F16, F32; },
            (Self::freg, _) => types! { f: F16, F32; d: F64; },
            (Self::vreg, _) => {
                if allow_experimental_reg {
                    types! {
                        lsx: F16, F32, F64,
                            VecI8(16), VecI16(8), VecI32(4), VecI64(2), VecF32(4), VecF64(2);
                    }
                } else {
                    &[]
                }
            }
            (Self::xreg, _) => {
                if allow_experimental_reg {
                    types! {
                        lasx: F16, F32, F64,
                            VecI8(16), VecI16(8), VecI32(4), VecI64(2), VecF32(4), VecF64(2),
                            VecI8(32), VecI16(16), VecI32(8), VecI64(4), VecF32(8), VecF64(4);
                    }
                } else {
                    &[]
                }
            }
            _ => unreachable!("unsupported register class"),
        }
    }
}

// The reserved registers are taken from <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/LoongArch/LoongArchRegisterInfo.cpp#79>
def_regs! {
    LoongArch LoongArchInlineAsmReg LoongArchInlineAsmRegClass {
        r1: reg = ["$r1","$ra"],
        r4: reg = ["$r4","$a0"],
        r5: reg = ["$r5","$a1"],
        r6: reg = ["$r6","$a2"],
        r7: reg = ["$r7","$a3"],
        r8: reg = ["$r8","$a4"],
        r9: reg = ["$r9","$a5"],
        r10: reg = ["$r10","$a6"],
        r11: reg = ["$r11","$a7"],
        r12: reg = ["$r12","$t0"],
        r13: reg = ["$r13","$t1"],
        r14: reg = ["$r14","$t2"],
        r15: reg = ["$r15","$t3"],
        r16: reg = ["$r16","$t4"],
        r17: reg = ["$r17","$t5"],
        r18: reg = ["$r18","$t6"],
        r19: reg = ["$r19","$t7"],
        r20: reg = ["$r20","$t8"],
        r23: reg = ["$r23","$s0"],
        r24: reg = ["$r24","$s1"],
        r25: reg = ["$r25","$s2"],
        r26: reg = ["$r26","$s3"],
        r27: reg = ["$r27","$s4"],
        r28: reg = ["$r28","$s5"],
        r29: reg = ["$r29","$s6"],
        r30: reg = ["$r30","$s7"],
        f0: freg = ["$f0","$fa0"],
        f1: freg = ["$f1","$fa1"],
        f2: freg = ["$f2","$fa2"],
        f3: freg = ["$f3","$fa3"],
        f4: freg = ["$f4","$fa4"],
        f5: freg = ["$f5","$fa5"],
        f6: freg = ["$f6","$fa6"],
        f7: freg = ["$f7","$fa7"],
        f8: freg = ["$f8","$ft0"],
        f9: freg = ["$f9","$ft1"],
        f10: freg = ["$f10","$ft2"],
        f11: freg = ["$f11","$ft3"],
        f12: freg = ["$f12","$ft4"],
        f13: freg = ["$f13","$ft5"],
        f14: freg = ["$f14","$ft6"],
        f15: freg = ["$f15","$ft7"],
        f16: freg = ["$f16","$ft8"],
        f17: freg = ["$f17","$ft9"],
        f18: freg = ["$f18","$ft10"],
        f19: freg = ["$f19","$ft11"],
        f20: freg = ["$f20","$ft12"],
        f21: freg = ["$f21","$ft13"],
        f22: freg = ["$f22","$ft14"],
        f23: freg = ["$f23","$ft15"],
        f24: freg = ["$f24","$fs0"],
        f25: freg = ["$f25","$fs1"],
        f26: freg = ["$f26","$fs2"],
        f27: freg = ["$f27","$fs3"],
        f28: freg = ["$f28","$fs4"],
        f29: freg = ["$f29","$fs5"],
        f30: freg = ["$f30","$fs6"],
        f31: freg = ["$f31","$fs7"],
        vr0: vreg = ["$vr0"],
        vr1: vreg = ["$vr1"],
        vr2: vreg = ["$vr2"],
        vr3: vreg = ["$vr3"],
        vr4: vreg = ["$vr4"],
        vr5: vreg = ["$vr5"],
        vr6: vreg = ["$vr6"],
        vr7: vreg = ["$vr7"],
        vr8: vreg = ["$vr8"],
        vr9: vreg = ["$vr9"],
        vr10: vreg = ["$vr10"],
        vr11: vreg = ["$vr11"],
        vr12: vreg = ["$vr12"],
        vr13: vreg = ["$vr13"],
        vr14: vreg = ["$vr14"],
        vr15: vreg = ["$vr15"],
        vr16: vreg = ["$vr16"],
        vr17: vreg = ["$vr17"],
        vr18: vreg = ["$vr18"],
        vr19: vreg = ["$vr19"],
        vr20: vreg = ["$vr20"],
        vr21: vreg = ["$vr21"],
        vr22: vreg = ["$vr22"],
        vr23: vreg = ["$vr23"],
        vr24: vreg = ["$vr24"],
        vr25: vreg = ["$vr25"],
        vr26: vreg = ["$vr26"],
        vr27: vreg = ["$vr27"],
        vr28: vreg = ["$vr28"],
        vr29: vreg = ["$vr29"],
        vr30: vreg = ["$vr30"],
        vr31: vreg = ["$vr31"],
        xr0: xreg = ["$xr0"],
        xr1: xreg = ["$xr1"],
        xr2: xreg = ["$xr2"],
        xr3: xreg = ["$xr3"],
        xr4: xreg = ["$xr4"],
        xr5: xreg = ["$xr5"],
        xr6: xreg = ["$xr6"],
        xr7: xreg = ["$xr7"],
        xr8: xreg = ["$xr8"],
        xr9: xreg = ["$xr9"],
        xr10: xreg = ["$xr10"],
        xr11: xreg = ["$xr11"],
        xr12: xreg = ["$xr12"],
        xr13: xreg = ["$xr13"],
        xr14: xreg = ["$xr14"],
        xr15: xreg = ["$xr15"],
        xr16: xreg = ["$xr16"],
        xr17: xreg = ["$xr17"],
        xr18: xreg = ["$xr18"],
        xr19: xreg = ["$xr19"],
        xr20: xreg = ["$xr20"],
        xr21: xreg = ["$xr21"],
        xr22: xreg = ["$xr22"],
        xr23: xreg = ["$xr23"],
        xr24: xreg = ["$xr24"],
        xr25: xreg = ["$xr25"],
        xr26: xreg = ["$xr26"],
        xr27: xreg = ["$xr27"],
        xr28: xreg = ["$xr28"],
        xr29: xreg = ["$xr29"],
        xr30: xreg = ["$xr30"],
        xr31: xreg = ["$xr31"],
        #error = ["$r0","$zero"] =>
            "constant zero cannot be used as an operand for inline asm",
        #error = ["$r2","$tp"] =>
            "reserved for TLS",
        #error = ["$r3","$sp"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["$r21"] =>
            "reserved by the ABI",
        #error = ["$r22","$fp"] =>
            "the frame pointer cannot be used as an operand for inline asm",
        #error = ["$r31","$s8"] =>
            "$r31 is used internally by LLVM and cannot be used as an operand for inline asm",
    }
}

impl LoongArchInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        out.write_str(self.name())
    }

    pub fn overlapping_regs(self, mut cb: impl FnMut(LoongArchInlineAsmReg)) {
        macro_rules! reg_conflicts {
            (
                $(
                    $f:ident : $v:ident : $x:ident
                ),*;
            ) => {
                match self {
                    $(
                        Self::$f | Self::$v | Self::$x => {
                            cb(Self::$f);
                            cb(Self::$v);
                            cb(Self::$x);
                        }
                    )*
                    r => cb(r),
                }
            };
        }

        reg_conflicts! {
            f0 : vr0 : xr0,
            f1 : vr1 : xr1,
            f2 : vr2 : xr2,
            f3 : vr3 : xr3,
            f4 : vr4 : xr4,
            f5 : vr5 : xr5,
            f6 : vr6 : xr6,
            f7 : vr7 : xr7,
            f8 : vr8 : xr8,
            f9 : vr9 : xr9,
            f10 : vr10 : xr10,
            f11 : vr11 : xr11,
            f12 : vr12 : xr12,
            f13 : vr13 : xr13,
            f14 : vr14 : xr14,
            f15 : vr15 : xr15,
            f16 : vr16 : xr16,
            f17 : vr17 : xr17,
            f18 : vr18 : xr18,
            f19 : vr19 : xr19,
            f20 : vr20 : xr20,
            f21 : vr21 : xr21,
            f22 : vr22 : xr22,
            f23 : vr23 : xr23,
            f24 : vr24 : xr24,
            f25 : vr25 : xr25,
            f26 : vr26 : xr26,
            f27 : vr27 : xr27,
            f28 : vr28 : xr28,
            f29 : vr29 : xr29,
            f30 : vr30 : xr30,
            f31 : vr31 : xr31;
        }
    }
}
