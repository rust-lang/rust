use std::fmt;

use rustc_data_structures::fx::FxIndexSet;
use rustc_span::Symbol;

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};
use crate::spec::{RelocModel, Target};

def_reg_class! {
    Sparc SparcInlineAsmRegClass {
        reg,
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
}
