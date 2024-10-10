use std::fmt;

use rustc_span::Symbol;

use super::{AArch64InlineAsmRegClass, InlineAsmArch};

def_regs! {
    Arm64EC Arm64ECInlineAsmReg AArch64 AArch64InlineAsmRegClass {
        x0: reg = ["x0", "w0"],
        x1: reg = ["x1", "w1"],
        x2: reg = ["x2", "w2"],
        x3: reg = ["x3", "w3"],
        x4: reg = ["x4", "w4"],
        x5: reg = ["x5", "w5"],
        x6: reg = ["x6", "w6"],
        x7: reg = ["x7", "w7"],
        x8: reg = ["x8", "w8"],
        x9: reg = ["x9", "w9"],
        x10: reg = ["x10", "w10"],
        x11: reg = ["x11", "w11"],
        x12: reg = ["x12", "w12"],
        x15: reg = ["x15", "w15"],
        x16: reg = ["x16", "w16"],
        x17: reg = ["x17", "w17"],
        x20: reg = ["x20", "w20"],
        x21: reg = ["x21", "w21"],
        x22: reg = ["x22", "w22"],
        x25: reg = ["x25", "w25"],
        x26: reg = ["x26", "w26"],
        x27: reg = ["x27", "w27"],
        x30: reg = ["x30", "w30", "lr", "wlr"],
        v0: vreg, vreg_low16 = ["v0", "b0", "h0", "s0", "d0", "q0"],
        v1: vreg, vreg_low16 = ["v1", "b1", "h1", "s1", "d1", "q1"],
        v2: vreg, vreg_low16 = ["v2", "b2", "h2", "s2", "d2", "q2"],
        v3: vreg, vreg_low16 = ["v3", "b3", "h3", "s3", "d3", "q3"],
        v4: vreg, vreg_low16 = ["v4", "b4", "h4", "s4", "d4", "q4"],
        v5: vreg, vreg_low16 = ["v5", "b5", "h5", "s5", "d5", "q5"],
        v6: vreg, vreg_low16 = ["v6", "b6", "h6", "s6", "d6", "q6"],
        v7: vreg, vreg_low16 = ["v7", "b7", "h7", "s7", "d7", "q7"],
        v8: vreg, vreg_low16 = ["v8", "b8", "h8", "s8", "d8", "q8"],
        v9: vreg, vreg_low16 = ["v9", "b9", "h9", "s9", "d9", "q9"],
        v10: vreg, vreg_low16 = ["v10", "b10", "h10", "s10", "d10", "q10"],
        v11: vreg, vreg_low16 = ["v11", "b11", "h11", "s11", "d11", "q11"],
        v12: vreg, vreg_low16 = ["v12", "b12", "h12", "s12", "d12", "q12"],
        v13: vreg, vreg_low16 = ["v13", "b13", "h13", "s13", "d13", "q13"],
        v14: vreg, vreg_low16 = ["v14", "b14", "h14", "s14", "d14", "q14"],
        v15: vreg, vreg_low16 = ["v15", "b15", "h15", "s15", "d15", "q15"],
        #error = ["x18", "w18"] =>
            "x18 is a reserved register on this target",
        #error = ["x19", "w19"] =>
            "x19 is used internally by LLVM and cannot be used as an operand for inline asm",
        #error = ["x29", "w29", "fp", "wfp"] =>
            "the frame pointer cannot be used as an operand for inline asm",
        #error = ["sp", "wsp"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["xzr", "wzr"] =>
            "the zero register cannot be used as an operand for inline asm",
        #error = [
            "x13", "w13",
            "x14", "w14",
            "x23", "w23",
            "x24", "w24",
            "x28", "w28",
            "v16", "b16", "h16", "s16", "d16", "q16",
            "v17", "b17", "h17", "s17", "d17", "q17",
            "v18", "b18", "h18", "s18", "d18", "q18",
            "v19", "b19", "h19", "s19", "d19", "q19",
            "v20", "b20", "h20", "s20", "d20", "q20",
            "v21", "b21", "h21", "s21", "d21", "q21",
            "v22", "b22", "h22", "s22", "d22", "q22",
            "v23", "b23", "h23", "s23", "d23", "q23",
            "v24", "b24", "h24", "s24", "d24", "q24",
            "v25", "b25", "h25", "s25", "d25", "q25",
            "v26", "b26", "h26", "s26", "d26", "q26",
            "v27", "b27", "h27", "s27", "d27", "q27",
            "v28", "b28", "h28", "s28", "d28", "q28",
            "v29", "b29", "h29", "s29", "d29", "q29",
            "v30", "b30", "h30", "s30", "d30", "q30",
            "v31", "b31", "h31", "s31", "d31", "q31"
        ] =>
            "x13, x14, x23, x24, x28, v16-v31 cannot be used for Arm64EC",
        #error = [
            "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
            "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
            "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
            "z30", "z31",
            "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9",
            "p10", "p11", "p12", "p13", "p14", "p15",
            "ffr"
        ] =>
            "SVE cannot be used for Arm64EC",
    }
}

impl Arm64ECInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        modifier: Option<char>,
    ) -> fmt::Result {
        let (prefix, index) = if let Some(index) = self.reg_index() {
            (modifier.unwrap_or('x'), index)
        } else if let Some(index) = self.vreg_index() {
            (modifier.unwrap_or('v'), index)
        } else {
            return out.write_str(self.name());
        };
        assert!(index < 32);
        write!(out, "{prefix}{index}")
    }

    /// If the register is an integer register then return its index.
    pub fn reg_index(self) -> Option<u32> {
        // Unlike `vreg_index`, we can't subtract `x0` to get the u32 because
        // `x13`, `x19`, `x29`, etc. are missing and the integer constants for the
        // `x0`..`x30` enum variants don't all match the register number. E.g. the
        // integer constant for `x12` is 12, but the constant for `x15` is 13.
        use Arm64ECInlineAsmReg::*;
        Some(match self {
            x0 => 0,
            x1 => 1,
            x2 => 2,
            x3 => 3,
            x4 => 4,
            x5 => 5,
            x6 => 6,
            x7 => 7,
            x8 => 8,
            x9 => 9,
            x10 => 10,
            x11 => 11,
            x12 => 12,
            // x13 is reserved
            // x14 is reserved
            x15 => 15,
            x16 => 16,
            x17 => 17,
            // x18 is reserved
            // x19 is reserved
            x20 => 20,
            x21 => 21,
            x22 => 22,
            // x23 is reserved
            // x24 is reserved
            x25 => 25,
            x26 => 26,
            x27 => 27,
            // x28 is reserved
            // x29 is reserved
            x30 => 30,
            _ => return None,
        })
    }

    /// If the register is a vector register then return its index.
    pub fn vreg_index(self) -> Option<u32> {
        use Arm64ECInlineAsmReg::*;
        if self as u32 >= v0 as u32 && self as u32 <= v15 as u32 {
            return Some(self as u32 - v0 as u32);
        }
        None
    }
}
