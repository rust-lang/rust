use std::fmt;

use rustc_data_structures::fx::FxIndexSet;
use rustc_span::{Symbol, sym};

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};
use crate::spec::{RelocModel, Target};

def_reg_class! {
    Sparc SparcInlineAsmRegClass {
        reg,
        freg,
        dreg,
        qreg,
        yreg,
    }
}

impl SparcInlineAsmRegClass {
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
            Self::reg => {
                if arch == InlineAsmArch::Sparc {
                    types! {
                        _: I8, I16, I32;
                        // FIXME: i64 is ok for g*/o* registers on SPARC-V8+ ("h" constraint in GCC),
                        //        but not yet supported in LLVM.
                        // v8plus: I64;
                    }
                } else {
                    types! { _: I8, I16, I32, I64; }
                }
            }
            Self::freg => types! { _: F32; },
            Self::dreg => types! { _: F64; },
            Self::qreg => types! { _: F128; },
            Self::yreg => &[],
        }
    }
}

fn reserved_g5(
    arch: InlineAsmArch,
    _reloc_model: RelocModel,
    _target_features: &FxIndexSet<Symbol>,
    _target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    if arch == InlineAsmArch::Sparc {
        // FIXME: Section 2.1.5 "Function Registers with Unassigned Roles" of the V8+ Technical
        // Specification says "%g5; no longer reserved for system software" [1], but LLVM always
        // reserves it on SPARC32 [2].
        // [1]: https://temlib.org/pub/SparcStation/Standards/V8plus.pdf
        // [2]: https://github.com/llvm/llvm-project/blob/llvmorg-19.1.0/llvm/lib/Target/Sparc/SparcRegisterInfo.cpp#L64-L66
        Err("g5 is reserved for system on SPARC32")
    } else {
        Ok(())
    }
}

fn v9_only(
    _arch: InlineAsmArch,
    _reloc_model: RelocModel,
    target_features: &FxIndexSet<Symbol>,
    _target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    // FIXME: This is the what GCC/LLVM currently use to limit access to upper-half registers, but
    // it's unclear whether this is the correct behaviour. See the discussion around
    // https://github.com/rust-lang/rust/pull/160949#discussion_r3806194355.
    if !target_features.contains(&sym::v9) {
        Err("floating point registers in the upper half can only be used on SPARCv9")
    } else {
        Ok(())
    }
}

def_regs! {
    Sparc SparcInlineAsmReg SparcInlineAsmRegClass {
        // FIXME:
        // - LLVM has reserve-{g,o,l,i}N feature to reserve each general-purpose registers.
        // - g2-g4 are reserved for application (optional in both LLVM and GCC, and GCC has -mno-app-regs option to reserve them).
        // There are currently no builtin targets that use them, but in the future they may need to
        // be supported via options similar to AArch64's -Z fixed-x18.
        r2: reg = ["r2", "g2"], // % reserved_g2
        r3: reg = ["r3", "g3"], // % reserved_g3
        r4: reg = ["r4", "g4"], // % reserved_g4
        r5: reg = ["r5", "g5"] % reserved_g5,
        r8: reg = ["r8", "o0"], // % reserved_o0
        r9: reg = ["r9", "o1"], // % reserved_o1
        r10: reg = ["r10", "o2"], // % reserved_o2
        r11: reg = ["r11", "o3"], // % reserved_o3
        r12: reg = ["r12", "o4"], // % reserved_o4
        r13: reg = ["r13", "o5"], // % reserved_o5
        r15: reg = ["r15", "o7"], // % reserved_o7
        r16: reg = ["r16", "l0"], // % reserved_l0
        r17: reg = ["r17", "l1"], // % reserved_l1
        r18: reg = ["r18", "l2"], // % reserved_l2
        r19: reg = ["r19", "l3"], // % reserved_l3
        r20: reg = ["r20", "l4"], // % reserved_l4
        r21: reg = ["r21", "l5"], // % reserved_l5
        r22: reg = ["r22", "l6"], // % reserved_l6
        r23: reg = ["r23", "l7"], // % reserved_l7
        r24: reg = ["r24", "i0"], // % reserved_i0
        r25: reg = ["r25", "i1"], // % reserved_i1
        r26: reg = ["r26", "i2"], // % reserved_i2
        r27: reg = ["r27", "i3"], // % reserved_i3
        r28: reg = ["r28", "i4"], // % reserved_i4
        r29: reg = ["r29", "i5"], // % reserved_i5
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
        f16: freg = ["f16"],
        f17: freg = ["f17"],
        f18: freg = ["f18"],
        f19: freg = ["f19"],
        f20: freg = ["f20"],
        f21: freg = ["f21"],
        f22: freg = ["f22"],
        f23: freg = ["f23"],
        f24: freg = ["f24"],
        f25: freg = ["f25"],
        f26: freg = ["f26"],
        f27: freg = ["f27"],
        f28: freg = ["f28"],
        f29: freg = ["f29"],
        f30: freg = ["f30"],
        f31: freg = ["f31"],
        d0: dreg = ["d0"],
        d2: dreg = ["d2"],
        d4: dreg = ["d4"],
        d6: dreg = ["d6"],
        d8: dreg = ["d8"],
        d10: dreg = ["d10"],
        d12: dreg = ["d12"],
        d14: dreg = ["d14"],
        d16: dreg = ["d16"],
        d18: dreg = ["d18"],
        d20: dreg = ["d20"],
        d22: dreg = ["d22"],
        d24: dreg = ["d24"],
        d26: dreg = ["d26"],
        d28: dreg = ["d28"],
        d30: dreg = ["d30"],
        d32: dreg = ["d32"] % v9_only,
        d34: dreg = ["d34"] % v9_only,
        d36: dreg = ["d36"] % v9_only,
        d38: dreg = ["d38"] % v9_only,
        d40: dreg = ["d40"] % v9_only,
        d42: dreg = ["d42"] % v9_only,
        d44: dreg = ["d44"] % v9_only,
        d46: dreg = ["d46"] % v9_only,
        d48: dreg = ["d48"] % v9_only,
        d50: dreg = ["d50"] % v9_only,
        d52: dreg = ["d52"] % v9_only,
        d54: dreg = ["d54"] % v9_only,
        d56: dreg = ["d56"] % v9_only,
        d58: dreg = ["d58"] % v9_only,
        d60: dreg = ["d60"] % v9_only,
        d62: dreg = ["d62"] % v9_only,
        q0: qreg = ["q0"],
        q4: qreg = ["q4"],
        q8: qreg = ["q8"],
        q12: qreg = ["q12"],
        q16: qreg = ["q16"],
        q20: qreg = ["q20"],
        q24: qreg = ["q24"],
        q28: qreg = ["q28"],
        q32: qreg = ["q32"] % v9_only,
        q36: qreg = ["q36"] % v9_only,
        q40: qreg = ["q40"] % v9_only,
        q44: qreg = ["q44"] % v9_only,
        q48: qreg = ["q48"] % v9_only,
        q52: qreg = ["q52"] % v9_only,
        q56: qreg = ["q56"] % v9_only,
        q60: qreg = ["q60"] % v9_only,
        y: yreg = ["y"],
        #error = ["r0", "g0"] =>
            "g0 is always zero and cannot be used as an operand for inline asm",
        // FIXME: %g1 is volatile in ABI, but used internally by LLVM.
        // https://github.com/llvm/llvm-project/blob/llvmorg-19.1.0/llvm/lib/Target/Sparc/SparcRegisterInfo.cpp#L55-L56
        // > FIXME: G1 reserved for now for large imm generation by frame code.
        #error = ["r1", "g1"] =>
            "reserved by LLVM and cannot be used as an operand for inline asm",
        #error = ["r6", "g6", "r7", "g7"] =>
            "reserved for system and cannot be used as an operand for inline asm",
        #error = ["sp", "r14", "o6"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["fp", "r30", "i6"] =>
            "the frame pointer cannot be used as an operand for inline asm",
        #error = ["r31", "i7"] =>
            "the return address register cannot be used as an operand for inline asm",
    }
}

impl SparcInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        write!(out, "%{}", self.name())
    }

    pub fn overlapping_regs(self, mut cb: impl FnMut(SparcInlineAsmReg)) {
        cb(self);

        macro_rules! reg_conflicts {
            (
                $(
                    $q:ident : $d0:ident $d1:ident : $f0:ident $f1:ident $f2:ident $f3:ident
                ),*;
                $(
                    $q_high:ident : $d0_high:ident $d1_high:ident
                ),*;
            ) => {
                match self {
                    $(
                        Self::$q => {
                            cb(Self::$d0);
                            cb(Self::$d1);
                            cb(Self::$f0);
                            cb(Self::$f1);
                            cb(Self::$f2);
                            cb(Self::$f3);
                        }
                        Self::$d0 => {
                            cb(Self::$q);
                            cb(Self::$f0);
                            cb(Self::$f1);
                        }
                        Self::$d1 => {
                            cb(Self::$q);
                            cb(Self::$f2);
                            cb(Self::$f3);
                        }
                        Self::$f0 | Self::$f1 => {
                            cb(Self::$q);
                            cb(Self::$d0);
                        }
                        Self::$f2 | Self::$f3 => {
                            cb(Self::$q);
                            cb(Self::$d1);
                        }
                    )*
                    $(
                        Self::$q_high => {
                            cb(Self::$d0_high);
                            cb(Self::$d1_high);
                        }
                        Self::$d0_high | Self::$d1_high => {
                            cb(Self::$q_high);
                        }
                    )*
                    _ => {},
                }
            };
        }

        // SPARC's floating-point register file is interesting in that it can be
        // viewed as 16 128-bit registers, 32 64-bit registers or 32 32-bit
        // registers. Because these views overlap, the registers of different
        // widths will conflict (e.g. d0 overlaps with f0 and f1, and q1
        // overlaps with d2 and d3).
        //
        // See section 3.1.2 of The SPARC Architecture Manual: Version 9 for details.
        reg_conflicts! {
            q0 : d0 d2 : f0 f1 f2 f3,
            q4 : d4 d6 : f4 f5 f6 f7,
            q8 : d8 d10 : f8 f9 f10 f11,
            q12 : d12 d14 : f12 f13 f14 f15,
            q16 : d16 d18 : f16 f17 f18 f19,
            q20 : d20 d22 : f20 f21 f22 f23,
            q24 : d24 d26 : f24 f25 f26 f27,
            q28 : d28 d30 : f28 f29 f30 f31;
            q32 : d32 d34,
            q36 : d36 d38,
            q40 : d40 d42,
            q44 : d44 d46,
            q48 : d48 d50,
            q52 : d52 d54,
            q56 : d56 d58,
            q60 : d60 d62;
        }
    }

    pub fn dreg_number(self) -> Option<u32> {
        if self >= Self::d0 && self <= Self::d62 {
            Some((self as u32 - Self::d0 as u32) * 2)
        } else {
            None
        }
    }

    pub fn qreg_number(self) -> Option<u32> {
        if self >= Self::q0 && self <= Self::q60 {
            Some((self as u32 - Self::q0 as u32) * 4)
        } else {
            None
        }
    }
}
