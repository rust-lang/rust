use std::fmt;

use rustc_data_structures::fx::FxIndexSet;
use rustc_span::Symbol;

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};
use crate::spec::{RelocModel, Target};

def_reg_class! {
    PowerPC PowerPCInlineAsmRegClass {
        reg,
        reg_nonzero,
        freg,
        vreg,
        vsreg,
        cr,
        ctr,
        lr,
        xer,
    }
}

impl PowerPCInlineAsmRegClass {
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
        arch: InlineAsmArch,
    ) -> &'static [(InlineAsmType, Option<Symbol>)] {
        match self {
            Self::reg | Self::reg_nonzero => {
                if arch == InlineAsmArch::PowerPC {
                    types! { _: I8, I16, I32; }
                } else {
                    types! { _: I8, I16, I32, I64; }
                }
            }
            Self::freg => types! { _: F32, F64; },
            // FIXME: vsx also supports integers?: https://github.com/rust-lang/rust/pull/131551#discussion_r1862535963
            Self::vreg => types! {
                altivec: VecI8(16), VecI16(8), VecI32(4), VecF32(4);
                vsx: F32, F64, VecI64(2), VecF64(2);
            },
            // VSX is a superset of altivec.
            Self::vsreg => types! {
                vsx: F32, F64, VecI8(16), VecI16(8), VecI32(4), VecI64(2), VecF32(4), VecF64(2);
            },
            Self::cr | Self::ctr | Self::lr | Self::xer => &[],
        }
    }
}

fn reserved_r13(
    arch: InlineAsmArch,
    _reloc_model: RelocModel,
    _target_features: &FxIndexSet<Symbol>,
    target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    if target.is_like_aix && arch == InlineAsmArch::PowerPC {
        Ok(())
    } else {
        Err("r13 is a reserved register on this target")
    }
}

fn reserved_r29(
    arch: InlineAsmArch,
    _reloc_model: RelocModel,
    _target_features: &FxIndexSet<Symbol>,
    _target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    if arch != InlineAsmArch::PowerPC {
        Ok(())
    } else {
        Err("r29 is used internally by LLVM and cannot be used as an operand for inline asm")
    }
}

fn reserved_v20to31(
    _arch: InlineAsmArch,
    _reloc_model: RelocModel,
    _target_features: &FxIndexSet<Symbol>,
    target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    if target.is_like_aix {
        match &*target.options.abi {
            "vec-default" => Err("v20-v31 (vs52-vs63) are reserved on vec-default ABI"),
            "vec-extabi" => Ok(()),
            _ => unreachable!("unrecognized AIX ABI"),
        }
    } else {
        Ok(())
    }
}

def_regs! {
    PowerPC PowerPCInlineAsmReg PowerPCInlineAsmRegClass {
        r0: reg = ["r0", "0"],
        r3: reg, reg_nonzero = ["r3", "3"],
        r4: reg, reg_nonzero = ["r4", "4"],
        r5: reg, reg_nonzero = ["r5", "5"],
        r6: reg, reg_nonzero = ["r6", "6"],
        r7: reg, reg_nonzero = ["r7", "7"],
        r8: reg, reg_nonzero = ["r8", "8"],
        r9: reg, reg_nonzero = ["r9", "9"],
        r10: reg, reg_nonzero = ["r10", "10"],
        r11: reg, reg_nonzero = ["r11", "11"],
        r12: reg, reg_nonzero = ["r12", "12"],
        r13: reg, reg_nonzero = ["r13", "13"] % reserved_r13,
        r14: reg, reg_nonzero = ["r14", "14"],
        r15: reg, reg_nonzero = ["r15", "15"],
        r16: reg, reg_nonzero = ["r16", "16"],
        r17: reg, reg_nonzero = ["r17", "17"],
        r18: reg, reg_nonzero = ["r18", "18"],
        r19: reg, reg_nonzero = ["r19", "19"],
        r20: reg, reg_nonzero = ["r20", "20"],
        r21: reg, reg_nonzero = ["r21", "21"],
        r22: reg, reg_nonzero = ["r22", "22"],
        r23: reg, reg_nonzero = ["r23", "23"],
        r24: reg, reg_nonzero = ["r24", "24"],
        r25: reg, reg_nonzero = ["r25", "25"],
        r26: reg, reg_nonzero = ["r26", "26"],
        r27: reg, reg_nonzero = ["r27", "27"],
        r28: reg, reg_nonzero = ["r28", "28"],
        r29: reg, reg_nonzero = ["r29", "29"] % reserved_r29,
        f0: freg = ["f0", "fr0"],
        f1: freg = ["f1", "fr1"],
        f2: freg = ["f2", "fr2"],
        f3: freg = ["f3", "fr3"],
        f4: freg = ["f4", "fr4"],
        f5: freg = ["f5", "fr5"],
        f6: freg = ["f6", "fr6"],
        f7: freg = ["f7", "fr7"],
        f8: freg = ["f8", "fr8"],
        f9: freg = ["f9", "fr9"],
        f10: freg = ["f10", "fr10"],
        f11: freg = ["f11", "fr11"],
        f12: freg = ["f12", "fr12"],
        f13: freg = ["f13", "fr13"],
        f14: freg = ["f14", "fr14"],
        f15: freg = ["f15", "fr15"],
        f16: freg = ["f16", "fr16"],
        f17: freg = ["f17", "fr17"],
        f18: freg = ["f18", "fr18"],
        f19: freg = ["f19", "fr19"],
        f20: freg = ["f20", "fr20"],
        f21: freg = ["f21", "fr21"],
        f22: freg = ["f22", "fr22"],
        f23: freg = ["f23", "fr23"],
        f24: freg = ["f24", "fr24"],
        f25: freg = ["f25", "fr25"],
        f26: freg = ["f26", "fr26"],
        f27: freg = ["f27", "fr27"],
        f28: freg = ["f28", "fr28"],
        f29: freg = ["f29", "fr29"],
        f30: freg = ["f30", "fr30"],
        f31: freg = ["f31", "fr31"],
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
        v20: vreg = ["v20"] % reserved_v20to31,
        v21: vreg = ["v21"] % reserved_v20to31,
        v22: vreg = ["v22"] % reserved_v20to31,
        v23: vreg = ["v23"] % reserved_v20to31,
        v24: vreg = ["v24"] % reserved_v20to31,
        v25: vreg = ["v25"] % reserved_v20to31,
        v26: vreg = ["v26"] % reserved_v20to31,
        v27: vreg = ["v27"] % reserved_v20to31,
        v28: vreg = ["v28"] % reserved_v20to31,
        v29: vreg = ["v29"] % reserved_v20to31,
        v30: vreg = ["v30"] % reserved_v20to31,
        v31: vreg = ["v31"] % reserved_v20to31,
        vs0: vsreg = ["vs0"],
        vs1: vsreg = ["vs1"],
        vs2: vsreg = ["vs2"],
        vs3: vsreg = ["vs3"],
        vs4: vsreg = ["vs4"],
        vs5: vsreg = ["vs5"],
        vs6: vsreg = ["vs6"],
        vs7: vsreg = ["vs7"],
        vs8: vsreg = ["vs8"],
        vs9: vsreg = ["vs9"],
        vs10: vsreg = ["vs10"],
        vs11: vsreg = ["vs11"],
        vs12: vsreg = ["vs12"],
        vs13: vsreg = ["vs13"],
        vs14: vsreg = ["vs14"],
        vs15: vsreg = ["vs15"],
        vs16: vsreg = ["vs16"],
        vs17: vsreg = ["vs17"],
        vs18: vsreg = ["vs18"],
        vs19: vsreg = ["vs19"],
        vs20: vsreg = ["vs20"],
        vs21: vsreg = ["vs21"],
        vs22: vsreg = ["vs22"],
        vs23: vsreg = ["vs23"],
        vs24: vsreg = ["vs24"],
        vs25: vsreg = ["vs25"],
        vs26: vsreg = ["vs26"],
        vs27: vsreg = ["vs27"],
        vs28: vsreg = ["vs28"],
        vs29: vsreg = ["vs29"],
        vs30: vsreg = ["vs30"],
        vs31: vsreg = ["vs31"],
        vs32: vsreg = ["vs32"],
        vs33: vsreg = ["vs33"],
        vs34: vsreg = ["vs34"],
        vs35: vsreg = ["vs35"],
        vs36: vsreg = ["vs36"],
        vs37: vsreg = ["vs37"],
        vs38: vsreg = ["vs38"],
        vs39: vsreg = ["vs39"],
        vs40: vsreg = ["vs40"],
        vs41: vsreg = ["vs41"],
        vs42: vsreg = ["vs42"],
        vs43: vsreg = ["vs43"],
        vs44: vsreg = ["vs44"],
        vs45: vsreg = ["vs45"],
        vs46: vsreg = ["vs46"],
        vs47: vsreg = ["vs47"],
        vs48: vsreg = ["vs48"],
        vs49: vsreg = ["vs49"],
        vs50: vsreg = ["vs50"],
        vs51: vsreg = ["vs51"],
        // vs52 - vs63 are aliases of v20-v31.
        vs52: vsreg = ["vs52"] % reserved_v20to31,
        vs53: vsreg = ["vs53"] % reserved_v20to31,
        vs54: vsreg = ["vs54"] % reserved_v20to31,
        vs55: vsreg = ["vs55"] % reserved_v20to31,
        vs56: vsreg = ["vs56"] % reserved_v20to31,
        vs57: vsreg = ["vs57"] % reserved_v20to31,
        vs58: vsreg = ["vs58"] % reserved_v20to31,
        vs59: vsreg = ["vs59"] % reserved_v20to31,
        vs60: vsreg = ["vs60"] % reserved_v20to31,
        vs61: vsreg = ["vs61"] % reserved_v20to31,
        vs62: vsreg = ["vs62"] % reserved_v20to31,
        vs63: vsreg = ["vs63"] % reserved_v20to31,
        cr: cr = ["cr"],
        cr0: cr = ["cr0"],
        cr1: cr = ["cr1"],
        cr2: cr = ["cr2"],
        cr3: cr = ["cr3"],
        cr4: cr = ["cr4"],
        cr5: cr = ["cr5"],
        cr6: cr = ["cr6"],
        cr7: cr = ["cr7"],
        ctr: ctr = ["ctr"],
        lr: lr = ["lr"],
        xer: xer = ["xer"],
        #error = ["r1", "1", "sp"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["r2", "2"] =>
            "r2 is a system reserved register and cannot be used as an operand for inline asm",
        #error = ["r30", "30"] =>
            "r30 is used internally by LLVM and cannot be used as an operand for inline asm",
        #error = ["r31", "31", "fp"] =>
            "the frame pointer cannot be used as an operand for inline asm",
        #error = ["vrsave"] =>
            "the vrsave register cannot be used as an operand for inline asm",
    }
}

impl PowerPCInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        macro_rules! do_emit {
            (
                $($(($reg:ident, $value:literal)),*;)*
            ) => {
                out.write_str(match self {
                    $($(Self::$reg => $value,)*)*
                })
            };
        }
        // Strip off the leading prefix.
        do_emit! {
            (r0, "0"), (r3, "3"), (r4, "4"), (r5, "5"), (r6, "6"), (r7, "7");
            (r8, "8"), (r9, "9"), (r10, "10"), (r11, "11"), (r12, "12"), (r13, "13"), (r14, "14"), (r15, "15");
            (r16, "16"), (r17, "17"), (r18, "18"), (r19, "19"), (r20, "20"), (r21, "21"), (r22, "22"), (r23, "23");
            (r24, "24"), (r25, "25"), (r26, "26"), (r27, "27"), (r28, "28"), (r29, "29");
            (f0, "0"), (f1, "1"), (f2, "2"), (f3, "3"), (f4, "4"), (f5, "5"), (f6, "6"), (f7, "7");
            (f8, "8"), (f9, "9"), (f10, "10"), (f11, "11"), (f12, "12"), (f13, "13"), (f14, "14"), (f15, "15");
            (f16, "16"), (f17, "17"), (f18, "18"), (f19, "19"), (f20, "20"), (f21, "21"), (f22, "22"), (f23, "23");
            (f24, "24"), (f25, "25"), (f26, "26"), (f27, "27"), (f28, "28"), (f29, "29"), (f30, "30"), (f31, "31");
            (v0, "0"), (v1, "1"), (v2, "2"), (v3, "3"), (v4, "4"), (v5, "5"), (v6, "6"), (v7, "7");
            (v8, "8"), (v9, "9"), (v10, "10"), (v11, "11"), (v12, "12"), (v13, "13"), (v14, "14"), (v15, "15");
            (v16, "16"), (v17, "17"), (v18, "18"), (v19, "19"), (v20, "20"), (v21, "21"), (v22, "22"), (v23, "23");
            (v24, "24"), (v25, "25"), (v26, "26"), (v27, "27"), (v28, "28"), (v29, "29"), (v30, "30"), (v31, "31");
            (vs0, "0"), (vs1, "1"), (vs2, "2"), (vs3, "3"), (vs4, "4"), (vs5, "5"), (vs6, "6"), (vs7, "7"),
            (vs8, "8"), (vs9, "9"), (vs10, "10"), (vs11, "11"), (vs12, "12"), (vs13, "13"), (vs14, "14"),
            (vs15, "15"), (vs16, "16"), (vs17, "17"), (vs18, "18"), (vs19, "19"), (vs20, "20"), (vs21, "21"),
            (vs22, "22"), (vs23, "23"), (vs24, "24"), (vs25, "25"), (vs26, "26"), (vs27, "27"), (vs28, "28"),
            (vs29, "29"), (vs30, "30"), (vs31, "31"), (vs32, "32"), (vs33, "33"), (vs34, "34"), (vs35, "35"),
            (vs36, "36"), (vs37, "37"), (vs38, "38"), (vs39, "39"), (vs40, "40"), (vs41, "41"), (vs42, "42"),
            (vs43, "43"), (vs44, "44"), (vs45, "45"), (vs46, "46"), (vs47, "47"), (vs48, "48"), (vs49, "49"),
            (vs50, "50"), (vs51, "51"), (vs52, "52"), (vs53, "53"), (vs54, "54"), (vs55, "55"), (vs56, "56"),
            (vs57, "57"), (vs58, "58"), (vs59, "59"), (vs60, "60"), (vs61, "61"), (vs62, "62"), (vs63, "63"),
            (cr, "cr");
            (cr0, "0"), (cr1, "1"), (cr2, "2"), (cr3, "3"), (cr4, "4"), (cr5, "5"), (cr6, "6"), (cr7, "7");
            (ctr, "ctr");
            (lr, "lr");
            (xer, "xer");
        }
    }

    pub fn overlapping_regs(self, mut cb: impl FnMut(PowerPCInlineAsmReg)) {
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
        reg_conflicts! {
            cr : cr0 cr1 cr2 cr3 cr4 cr5 cr6 cr7,
            // f0-f31 overlap half of each of vs0-vs32.
            vs0 : f0,
            vs1 : f1,
            vs2 : f2,
            vs3 : f3,
            vs4 : f4,
            vs5 : f5,
            vs6 : f6,
            vs7 : f7,
            vs8 : f8,
            vs9 : f9,
            vs10 : f10,
            vs11 : f11,
            vs12 : f12,
            vs13 : f13,
            vs14 : f14,
            vs15 : f15,
            vs16 : f16,
            vs17 : f17,
            vs18 : f18,
            vs19 : f19,
            vs20 : f20,
            vs21 : f21,
            vs22 : f22,
            vs23 : f23,
            vs24 : f24,
            vs25 : f25,
            vs26 : f26,
            vs27 : f27,
            vs28 : f28,
            vs29 : f29,
            vs30 : f30,
            vs31 : f31,
            // vs32-v63 are aliases of v0-v31
            vs32 : v0,
            vs33 : v1,
            vs34 : v2,
            vs35 : v3,
            vs36 : v4,
            vs37 : v5,
            vs38 : v6,
            vs39 : v7,
            vs40 : v8,
            vs41 : v9,
            vs42 : v10,
            vs43 : v11,
            vs44 : v12,
            vs45 : v13,
            vs46 : v14,
            vs47 : v15,
            vs48 : v16,
            vs49 : v17,
            vs50 : v18,
            vs51 : v19,
            vs52 : v20,
            vs53 : v21,
            vs54 : v22,
            vs55 : v23,
            vs56 : v24,
            vs57 : v25,
            vs58 : v26,
            vs59 : v27,
            vs60 : v28,
            vs61 : v29,
            vs62 : v30,
            vs63 : v31;
        }
        // For more detail on how vsx, vmx (altivec), fpr, and mma registers overlap
        // see OpenPOWER ISA 3.1C, Book I, Section 7.2.1.1 through 7.2.1.3.
        //
        // https://files.openpower.foundation/s/9izgC5Rogi5Ywmm
    }
}
