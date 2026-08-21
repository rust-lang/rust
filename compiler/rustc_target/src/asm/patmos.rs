use std::fmt;

use rustc_span::Symbol;

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};

def_reg_class! {
    Patmos PatmosInlineAsmRegClass {
        reg,
        sreg,
    }
}

def_regs! {
    Patmos PatmosInlineAsmReg PatmosInlineAsmRegClass {
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
        r19: reg = ["r19"],
        r20: reg = ["r20"],
        r21: reg = ["r21"],
        r22: reg = ["r22"],
        r23: reg = ["r23"],
        r24: reg = ["r24"],
        r25: reg = ["r25"],
        r26: reg = ["r26"],
        r27: reg = ["r27"],
        r28: reg = ["r28"],

        s1: sreg = ["s1"],
        s2: sreg = ["s2"],
        s3: sreg = ["s3"],
        s4: sreg = ["s4"],
        s5: sreg = ["s5"],
        s6: sreg = ["s6"],
        s7: sreg = ["s7"],
        s8: sreg = ["s8"],
        s9: sreg = ["s9"],
        s10: sreg = ["s10"],
        s11: sreg = ["s11"],
        s12: sreg = ["s12"],
        s13: sreg = ["s13"],
        s14: sreg = ["s14"],
        s15: sreg = ["s15"],

        #error = ["r0"] =>
            "constant zero cannot be used as an operand for inline asm",
        #error = ["r29", "rtr"] =>
            "r29 is a reserved temporary register and cannot be used as an operand for inline asm",
        #error = ["r30", "rfp"] =>
            "the frame pointer cannot be used as an operand for inline asm",
        #error = ["r31", "rsp"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["s0"] =>
            "s0 is a constant zero register and cannot be used as an operand for inline asm",
    }
}

impl PatmosInlineAsmRegClass {
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
        types! { _: I8, I16, I32; }
    }
}

impl PatmosInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        out.write_str(self.name())
    }
}
