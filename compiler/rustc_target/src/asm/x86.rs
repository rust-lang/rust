use super::{InlineAsmArch, InlineAsmType};
use crate::spec::Target;
use rustc_data_structures::stable_set::FxHashSet;
use rustc_macros::HashStable_Generic;
use rustc_span::Symbol;
use std::fmt;

def_reg_class! {
    X86 X86InlineAsmRegClass {
        reg,
        reg_abcd,
        reg_byte,
        xmm_reg,
        ymm_reg,
        zmm_reg,
        kreg,
        mmx_reg,
        x87_reg,
    }
}

impl X86InlineAsmRegClass {
    pub fn valid_modifiers(self, arch: super::InlineAsmArch) -> &'static [char] {
        match self {
            Self::reg => {
                if arch == InlineAsmArch::X86_64 {
                    &['l', 'x', 'e', 'r']
                } else {
                    &['x', 'e']
                }
            }
            Self::reg_abcd => {
                if arch == InlineAsmArch::X86_64 {
                    &['l', 'h', 'x', 'e', 'r']
                } else {
                    &['l', 'h', 'x', 'e']
                }
            }
            Self::reg_byte => &[],
            Self::xmm_reg | Self::ymm_reg | Self::zmm_reg => &['x', 'y', 'z'],
            Self::kreg => &[],
            Self::mmx_reg | Self::x87_reg => &[],
        }
    }

    pub fn suggest_class(self, _arch: InlineAsmArch, ty: InlineAsmType) -> Option<Self> {
        match self {
            Self::reg | Self::reg_abcd if ty.size().bits() == 8 => Some(Self::reg_byte),
            _ => None,
        }
    }

    pub fn suggest_modifier(
        self,
        arch: InlineAsmArch,
        ty: InlineAsmType,
    ) -> Option<(char, &'static str)> {
        match self {
            Self::reg => match ty.size().bits() {
                16 => Some(('x', "ax")),
                32 if arch == InlineAsmArch::X86_64 => Some(('e', "eax")),
                _ => None,
            },
            Self::reg_abcd => match ty.size().bits() {
                16 => Some(('x', "ax")),
                32 if arch == InlineAsmArch::X86_64 => Some(('e', "eax")),
                _ => None,
            },
            Self::reg_byte => None,
            Self::xmm_reg => None,
            Self::ymm_reg => match ty.size().bits() {
                256 => None,
                _ => Some(('x', "xmm0")),
            },
            Self::zmm_reg => match ty.size().bits() {
                512 => None,
                256 => Some(('y', "ymm0")),
                _ => Some(('x', "xmm0")),
            },
            Self::kreg => None,
            Self::mmx_reg | Self::x87_reg => None,
        }
    }

    pub fn default_modifier(self, arch: InlineAsmArch) -> Option<(char, &'static str)> {
        match self {
            Self::reg | Self::reg_abcd => {
                if arch == InlineAsmArch::X86_64 {
                    Some(('r', "rax"))
                } else {
                    Some(('e', "eax"))
                }
            }
            Self::reg_byte => None,
            Self::xmm_reg => Some(('x', "xmm0")),
            Self::ymm_reg => Some(('y', "ymm0")),
            Self::zmm_reg => Some(('z', "zmm0")),
            Self::kreg => None,
            Self::mmx_reg | Self::x87_reg => None,
        }
    }

    pub fn supported_types(
        self,
        arch: InlineAsmArch,
    ) -> &'static [(InlineAsmType, Option<Symbol>)] {
        match self {
            Self::reg | Self::reg_abcd => {
                if arch == InlineAsmArch::X86_64 {
                    types! { _: I16, I32, I64, F32, F64; }
                } else {
                    types! { _: I16, I32, F32; }
                }
            }
            Self::reg_byte => types! { _: I8; },
            Self::xmm_reg => types! {
                sse: I32, I64, F32, F64,
                  VecI8(16), VecI16(8), VecI32(4), VecI64(2), VecF32(4), VecF64(2);
            },
            Self::ymm_reg => types! {
                avx: I32, I64, F32, F64,
                    VecI8(16), VecI16(8), VecI32(4), VecI64(2), VecF32(4), VecF64(2),
                    VecI8(32), VecI16(16), VecI32(8), VecI64(4), VecF32(8), VecF64(4);
            },
            Self::zmm_reg => types! {
                avx512f: I32, I64, F32, F64,
                    VecI8(16), VecI16(8), VecI32(4), VecI64(2), VecF32(4), VecF64(2),
                    VecI8(32), VecI16(16), VecI32(8), VecI64(4), VecF32(8), VecF64(4),
                    VecI8(64), VecI16(32), VecI32(16), VecI64(8), VecF32(16), VecF64(8);
            },
            Self::kreg => types! {
                avx512f: I8, I16;
                avx512bw: I32, I64;
            },
            Self::mmx_reg | Self::x87_reg => &[],
        }
    }
}

fn x86_64_only(
    arch: InlineAsmArch,
    _target_features: &FxHashSet<Symbol>,
    _target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    match arch {
        InlineAsmArch::X86 => Err("register is only available on x86_64"),
        InlineAsmArch::X86_64 => Ok(()),
        _ => unreachable!(),
    }
}

fn high_byte(
    arch: InlineAsmArch,
    _target_features: &FxHashSet<Symbol>,
    _target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    match arch {
        InlineAsmArch::X86_64 => Err("high byte registers cannot be used as an operand on x86_64"),
        _ => Ok(()),
    }
}

fn rbx_reserved(
    arch: InlineAsmArch,
    _target_features: &FxHashSet<Symbol>,
    _target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    match arch {
        InlineAsmArch::X86 => Ok(()),
        InlineAsmArch::X86_64 => {
            Err("rbx is used internally by LLVM and cannot be used as an operand for inline asm")
        }
        _ => unreachable!(),
    }
}

fn esi_reserved(
    arch: InlineAsmArch,
    _target_features: &FxHashSet<Symbol>,
    _target: &Target,
    _is_clobber: bool,
) -> Result<(), &'static str> {
    match arch {
        InlineAsmArch::X86 => {
            Err("esi is used internally by LLVM and cannot be used as an operand for inline asm")
        }
        InlineAsmArch::X86_64 => Ok(()),
        _ => unreachable!(),
    }
}

def_regs! {
    X86 X86InlineAsmReg X86InlineAsmRegClass {
        ax: reg, reg_abcd = ["ax", "eax", "rax"],
        bx: reg, reg_abcd = ["bx", "ebx", "rbx"] % rbx_reserved,
        cx: reg, reg_abcd = ["cx", "ecx", "rcx"],
        dx: reg, reg_abcd = ["dx", "edx", "rdx"],
        si: reg = ["si", "esi", "rsi"] % esi_reserved,
        di: reg = ["di", "edi", "rdi"],
        r8: reg = ["r8", "r8w", "r8d"] % x86_64_only,
        r9: reg = ["r9", "r9w", "r9d"] % x86_64_only,
        r10: reg = ["r10", "r10w", "r10d"] % x86_64_only,
        r11: reg = ["r11", "r11w", "r11d"] % x86_64_only,
        r12: reg = ["r12", "r12w", "r12d"] % x86_64_only,
        r13: reg = ["r13", "r13w", "r13d"] % x86_64_only,
        r14: reg = ["r14", "r14w", "r14d"] % x86_64_only,
        r15: reg = ["r15", "r15w", "r15d"] % x86_64_only,
        al: reg_byte = ["al"],
        ah: reg_byte = ["ah"] % high_byte,
        bl: reg_byte = ["bl"],
        bh: reg_byte = ["bh"] % high_byte,
        cl: reg_byte = ["cl"],
        ch: reg_byte = ["ch"] % high_byte,
        dl: reg_byte = ["dl"],
        dh: reg_byte = ["dh"] % high_byte,
        sil: reg_byte = ["sil"] % x86_64_only,
        dil: reg_byte = ["dil"] % x86_64_only,
        r8b: reg_byte = ["r8b"] % x86_64_only,
        r9b: reg_byte = ["r9b"] % x86_64_only,
        r10b: reg_byte = ["r10b"] % x86_64_only,
        r11b: reg_byte = ["r11b"] % x86_64_only,
        r12b: reg_byte = ["r12b"] % x86_64_only,
        r13b: reg_byte = ["r13b"] % x86_64_only,
        r14b: reg_byte = ["r14b"] % x86_64_only,
        r15b: reg_byte = ["r15b"] % x86_64_only,
        xmm0: xmm_reg = ["xmm0"],
        xmm1: xmm_reg = ["xmm1"],
        xmm2: xmm_reg = ["xmm2"],
        xmm3: xmm_reg = ["xmm3"],
        xmm4: xmm_reg = ["xmm4"],
        xmm5: xmm_reg = ["xmm5"],
        xmm6: xmm_reg = ["xmm6"],
        xmm7: xmm_reg = ["xmm7"],
        xmm8: xmm_reg = ["xmm8"] % x86_64_only,
        xmm9: xmm_reg = ["xmm9"] % x86_64_only,
        xmm10: xmm_reg = ["xmm10"] % x86_64_only,
        xmm11: xmm_reg = ["xmm11"] % x86_64_only,
        xmm12: xmm_reg = ["xmm12"] % x86_64_only,
        xmm13: xmm_reg = ["xmm13"] % x86_64_only,
        xmm14: xmm_reg = ["xmm14"] % x86_64_only,
        xmm15: xmm_reg = ["xmm15"] % x86_64_only,
        ymm0: ymm_reg = ["ymm0"],
        ymm1: ymm_reg = ["ymm1"],
        ymm2: ymm_reg = ["ymm2"],
        ymm3: ymm_reg = ["ymm3"],
        ymm4: ymm_reg = ["ymm4"],
        ymm5: ymm_reg = ["ymm5"],
        ymm6: ymm_reg = ["ymm6"],
        ymm7: ymm_reg = ["ymm7"],
        ymm8: ymm_reg = ["ymm8"] % x86_64_only,
        ymm9: ymm_reg = ["ymm9"] % x86_64_only,
        ymm10: ymm_reg = ["ymm10"] % x86_64_only,
        ymm11: ymm_reg = ["ymm11"] % x86_64_only,
        ymm12: ymm_reg = ["ymm12"] % x86_64_only,
        ymm13: ymm_reg = ["ymm13"] % x86_64_only,
        ymm14: ymm_reg = ["ymm14"] % x86_64_only,
        ymm15: ymm_reg = ["ymm15"] % x86_64_only,
        zmm0: zmm_reg = ["zmm0"],
        zmm1: zmm_reg = ["zmm1"],
        zmm2: zmm_reg = ["zmm2"],
        zmm3: zmm_reg = ["zmm3"],
        zmm4: zmm_reg = ["zmm4"],
        zmm5: zmm_reg = ["zmm5"],
        zmm6: zmm_reg = ["zmm6"],
        zmm7: zmm_reg = ["zmm7"],
        zmm8: zmm_reg = ["zmm8"] % x86_64_only,
        zmm9: zmm_reg = ["zmm9"] % x86_64_only,
        zmm10: zmm_reg = ["zmm10"] % x86_64_only,
        zmm11: zmm_reg = ["zmm11"] % x86_64_only,
        zmm12: zmm_reg = ["zmm12"] % x86_64_only,
        zmm13: zmm_reg = ["zmm13"] % x86_64_only,
        zmm14: zmm_reg = ["zmm14"] % x86_64_only,
        zmm15: zmm_reg = ["zmm15"] % x86_64_only,
        zmm16: zmm_reg = ["zmm16", "xmm16", "ymm16"] % x86_64_only,
        zmm17: zmm_reg = ["zmm17", "xmm17", "ymm17"] % x86_64_only,
        zmm18: zmm_reg = ["zmm18", "xmm18", "ymm18"] % x86_64_only,
        zmm19: zmm_reg = ["zmm19", "xmm19", "ymm19"] % x86_64_only,
        zmm20: zmm_reg = ["zmm20", "xmm20", "ymm20"] % x86_64_only,
        zmm21: zmm_reg = ["zmm21", "xmm21", "ymm21"] % x86_64_only,
        zmm22: zmm_reg = ["zmm22", "xmm22", "ymm22"] % x86_64_only,
        zmm23: zmm_reg = ["zmm23", "xmm23", "ymm23"] % x86_64_only,
        zmm24: zmm_reg = ["zmm24", "xmm24", "ymm24"] % x86_64_only,
        zmm25: zmm_reg = ["zmm25", "xmm25", "ymm25"] % x86_64_only,
        zmm26: zmm_reg = ["zmm26", "xmm26", "ymm26"] % x86_64_only,
        zmm27: zmm_reg = ["zmm27", "xmm27", "ymm27"] % x86_64_only,
        zmm28: zmm_reg = ["zmm28", "xmm28", "ymm28"] % x86_64_only,
        zmm29: zmm_reg = ["zmm29", "xmm29", "ymm29"] % x86_64_only,
        zmm30: zmm_reg = ["zmm30", "xmm30", "ymm30"] % x86_64_only,
        zmm31: zmm_reg = ["zmm31", "xmm31", "ymm31"] % x86_64_only,
        k1: kreg = ["k1"],
        k2: kreg = ["k2"],
        k3: kreg = ["k3"],
        k4: kreg = ["k4"],
        k5: kreg = ["k5"],
        k6: kreg = ["k6"],
        k7: kreg = ["k7"],
        mm0: mmx_reg = ["mm0"],
        mm1: mmx_reg = ["mm1"],
        mm2: mmx_reg = ["mm2"],
        mm3: mmx_reg = ["mm3"],
        mm4: mmx_reg = ["mm4"],
        mm5: mmx_reg = ["mm5"],
        mm6: mmx_reg = ["mm6"],
        mm7: mmx_reg = ["mm7"],
        st0: x87_reg = ["st(0)", "st"],
        st1: x87_reg = ["st(1)"],
        st2: x87_reg = ["st(2)"],
        st3: x87_reg = ["st(3)"],
        st4: x87_reg = ["st(4)"],
        st5: x87_reg = ["st(5)"],
        st6: x87_reg = ["st(6)"],
        st7: x87_reg = ["st(7)"],
        #error = ["bp", "bpl", "ebp", "rbp"] =>
            "the frame pointer cannot be used as an operand for inline asm",
        #error = ["sp", "spl", "esp", "rsp"] =>
            "the stack pointer cannot be used as an operand for inline asm",
        #error = ["ip", "eip", "rip"] =>
            "the instruction pointer cannot be used as an operand for inline asm",
        #error = ["k0"] =>
            "the k0 AVX mask register cannot be used as an operand for inline asm",
    }
}

impl X86InlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        arch: InlineAsmArch,
        modifier: Option<char>,
    ) -> fmt::Result {
        let reg_default_modifier = match arch {
            InlineAsmArch::X86 => 'e',
            InlineAsmArch::X86_64 => 'r',
            _ => unreachable!(),
        };
        if self as u32 <= Self::dx as u32 {
            let root = ['a', 'b', 'c', 'd'][self as usize - Self::ax as usize];
            match modifier.unwrap_or(reg_default_modifier) {
                'l' => write!(out, "{}l", root),
                'h' => write!(out, "{}h", root),
                'x' => write!(out, "{}x", root),
                'e' => write!(out, "e{}x", root),
                'r' => write!(out, "r{}x", root),
                _ => unreachable!(),
            }
        } else if self as u32 <= Self::di as u32 {
            let root = self.name();
            match modifier.unwrap_or(reg_default_modifier) {
                'l' => write!(out, "{}l", root),
                'x' => write!(out, "{}", root),
                'e' => write!(out, "e{}", root),
                'r' => write!(out, "r{}", root),
                _ => unreachable!(),
            }
        } else if self as u32 <= Self::r15 as u32 {
            let root = self.name();
            match modifier.unwrap_or(reg_default_modifier) {
                'l' => write!(out, "{}b", root),
                'x' => write!(out, "{}w", root),
                'e' => write!(out, "{}d", root),
                'r' => out.write_str(root),
                _ => unreachable!(),
            }
        } else if self as u32 <= Self::r15b as u32 {
            out.write_str(self.name())
        } else if self as u32 <= Self::xmm15 as u32 {
            let prefix = modifier.unwrap_or('x');
            let index = self as u32 - Self::xmm0 as u32;
            write!(out, "{}{}", prefix, index)
        } else if self as u32 <= Self::ymm15 as u32 {
            let prefix = modifier.unwrap_or('y');
            let index = self as u32 - Self::ymm0 as u32;
            write!(out, "{}{}", prefix, index)
        } else if self as u32 <= Self::zmm31 as u32 {
            let prefix = modifier.unwrap_or('z');
            let index = self as u32 - Self::zmm0 as u32;
            write!(out, "{}{}", prefix, index)
        } else {
            out.write_str(self.name())
        }
    }

    pub fn overlapping_regs(self, mut cb: impl FnMut(X86InlineAsmReg)) {
        macro_rules! reg_conflicts {
            (
                $(
                    $w:ident : $l:ident $h:ident
                ),*;
                $(
                    $w2:ident : $l2:ident
                ),*;
                $(
                    $x:ident : $y:ident : $z:ident
                ),*;
            ) => {
                match self {
                    $(
                        Self::$w => {
                            cb(Self::$w);
                            cb(Self::$l);
                            cb(Self::$h);
                        }
                        Self::$l => {
                            cb(Self::$w);
                            cb(Self::$l);
                        }
                        Self::$h => {
                            cb(Self::$w);
                            cb(Self::$h);
                        }
                    )*
                    $(
                        Self::$w2 | Self::$l2 => {
                            cb(Self::$w2);
                            cb(Self::$l2);
                        }
                    )*
                    $(
                        Self::$x | Self::$y | Self::$z => {
                            cb(Self::$x);
                            cb(Self::$y);
                            cb(Self::$z);
                        }
                    )*
                    r => cb(r),
                }
            };
        }

        // XMM*, YMM* and ZMM* are all different views of the same register.
        //
        // See section 15.5 of the combined Intel® 64 and IA-32 Architectures
        // Software Developer’s Manual for more details.
        //
        // We don't need to specify conflicts for [x,y,z]mm[16-31] since these
        // registers are only available with AVX-512, so we just specify them
        // as aliases directly.
        reg_conflicts! {
            ax : al ah,
            bx : bl bh,
            cx : cl ch,
            dx : dl dh;
            si : sil,
            di : dil,
            r8 : r8b,
            r9 : r9b,
            r10 : r10b,
            r11 : r11b,
            r12 : r12b,
            r13 : r13b,
            r14 : r14b,
            r15 : r15b;
            xmm0 : ymm0 : zmm0,
            xmm1 : ymm1 : zmm1,
            xmm2 : ymm2 : zmm2,
            xmm3 : ymm3 : zmm3,
            xmm4 : ymm4 : zmm4,
            xmm5 : ymm5 : zmm5,
            xmm6 : ymm6 : zmm6,
            xmm7 : ymm7 : zmm7,
            xmm8 : ymm8 : zmm8,
            xmm9 : ymm9 : zmm9,
            xmm10 : ymm10 : zmm10,
            xmm11 : ymm11 : zmm11,
            xmm12 : ymm12 : zmm12,
            xmm13 : ymm13 : zmm13,
            xmm14 : ymm14 : zmm14,
            xmm15 : ymm15 : zmm15;
        }
    }
}
