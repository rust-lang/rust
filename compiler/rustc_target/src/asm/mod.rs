use crate::spec::Target;
use crate::{abi::Size, spec::RelocModel};
use rustc_data_structures::fx::{FxHashMap, FxIndexSet};
use rustc_macros::HashStable_Generic;
use rustc_span::Symbol;
use std::fmt;
use std::str::FromStr;

macro_rules! def_reg_class {
    ($arch:ident $arch_regclass:ident {
        $(
            $class:ident,
        )*
    }) => {
        #[derive(Copy, Clone, Encodable, Decodable, Debug, Eq, PartialEq, PartialOrd, Hash, HashStable_Generic)]
        #[allow(non_camel_case_types)]
        pub enum $arch_regclass {
            $($class,)*
        }

        impl $arch_regclass {
            pub fn name(self) -> rustc_span::Symbol {
                match self {
                    $(Self::$class => rustc_span::symbol::sym::$class,)*
                }
            }

            pub fn parse(name: rustc_span::Symbol) -> Result<Self, &'static str> {
                match name {
                    $(
                        rustc_span::sym::$class => Ok(Self::$class),
                    )*
                    _ => Err("unknown register class"),
                }
            }
        }

        pub(super) fn regclass_map() -> rustc_data_structures::fx::FxHashMap<
            super::InlineAsmRegClass,
            rustc_data_structures::fx::FxIndexSet<super::InlineAsmReg>,
        > {
            use rustc_data_structures::fx::FxHashMap;
            use rustc_data_structures::fx::FxIndexSet;
            use super::InlineAsmRegClass;
            let mut map = FxHashMap::default();
            $(
                map.insert(InlineAsmRegClass::$arch($arch_regclass::$class), FxIndexSet::default());
            )*
            map
        }
    }
}

macro_rules! def_regs {
    ($arch:ident $arch_reg:ident $arch_regclass:ident {
        $(
            $reg:ident: $class:ident $(, $extra_class:ident)* = [$reg_name:literal $(, $alias:literal)*] $(% $filter:ident)?,
        )*
        $(
            #error = [$($bad_reg:literal),+] => $error:literal,
        )*
    }) => {
        #[allow(unreachable_code)]
        #[derive(Copy, Clone, Encodable, Decodable, Debug, Eq, PartialEq, PartialOrd, Hash, HashStable_Generic)]
        #[allow(non_camel_case_types)]
        pub enum $arch_reg {
            $($reg,)*
        }

        impl $arch_reg {
            pub fn name(self) -> &'static str {
                match self {
                    $(Self::$reg => $reg_name,)*
                }
            }

            pub fn reg_class(self) -> $arch_regclass {
                match self {
                    $(Self::$reg => $arch_regclass::$class,)*
                }
            }

            pub fn parse(name: &str) -> Result<Self, &'static str> {
                match name {
                    $(
                        $($alias)|* | $reg_name => Ok(Self::$reg),
                    )*
                    $(
                        $($bad_reg)|* => Err($error),
                    )*
                    _ => Err("unknown register"),
                }
            }

            pub fn validate(self,
                _arch: super::InlineAsmArch,
                _reloc_model: crate::spec::RelocModel,
                _target_features: &rustc_data_structures::fx::FxIndexSet<Symbol>,
                _target: &crate::spec::Target,
                _is_clobber: bool,
            ) -> Result<(), &'static str> {
                match self {
                    $(
                        Self::$reg => {
                            $($filter(
                                _arch,
                                _reloc_model,
                                _target_features,
                                _target,
                                _is_clobber
                            )?;)?
                            Ok(())
                        }
                    )*
                }
            }
        }

        pub(super) fn fill_reg_map(
            _arch: super::InlineAsmArch,
            _reloc_model: crate::spec::RelocModel,
            _target_features: &rustc_data_structures::fx::FxIndexSet<Symbol>,
            _target: &crate::spec::Target,
            _map: &mut rustc_data_structures::fx::FxHashMap<
                super::InlineAsmRegClass,
                rustc_data_structures::fx::FxIndexSet<super::InlineAsmReg>,
            >,
        ) {
            #[allow(unused_imports)]
            use super::{InlineAsmReg, InlineAsmRegClass};
            $(
                if $($filter(_arch, _reloc_model, _target_features, _target, false).is_ok() &&)? true {
                    if let Some(set) = _map.get_mut(&InlineAsmRegClass::$arch($arch_regclass::$class)) {
                        set.insert(InlineAsmReg::$arch($arch_reg::$reg));
                    }
                    $(
                        if let Some(set) = _map.get_mut(&InlineAsmRegClass::$arch($arch_regclass::$extra_class)) {
                            set.insert(InlineAsmReg::$arch($arch_reg::$reg));
                        }
                    )*
                }
            )*
        }
    }
}

macro_rules! types {
    (
        $(_ : $($ty:expr),+;)?
        $($feature:ident: $($ty2:expr),+;)*
    ) => {
        {
            use super::InlineAsmType::*;
            &[
                $($(
                    ($ty, None),
                )*)?
                $($(
                    ($ty2, Some(rustc_span::sym::$feature)),
                )*)*
            ]
        }
    };
}

mod aarch64;
mod arm;
mod avr;
mod bpf;
mod hexagon;
mod mips;
mod msp430;
mod nvptx;
mod powerpc;
mod riscv;
mod s390x;
mod spirv;
mod wasm;
mod x86;

pub use aarch64::{AArch64InlineAsmReg, AArch64InlineAsmRegClass};
pub use arm::{ArmInlineAsmReg, ArmInlineAsmRegClass};
pub use avr::{AvrInlineAsmReg, AvrInlineAsmRegClass};
pub use bpf::{BpfInlineAsmReg, BpfInlineAsmRegClass};
pub use hexagon::{HexagonInlineAsmReg, HexagonInlineAsmRegClass};
pub use mips::{MipsInlineAsmReg, MipsInlineAsmRegClass};
pub use msp430::{Msp430InlineAsmReg, Msp430InlineAsmRegClass};
pub use nvptx::{NvptxInlineAsmReg, NvptxInlineAsmRegClass};
pub use powerpc::{PowerPCInlineAsmReg, PowerPCInlineAsmRegClass};
pub use riscv::{RiscVInlineAsmReg, RiscVInlineAsmRegClass};
pub use s390x::{S390xInlineAsmReg, S390xInlineAsmRegClass};
pub use spirv::{SpirVInlineAsmReg, SpirVInlineAsmRegClass};
pub use wasm::{WasmInlineAsmReg, WasmInlineAsmRegClass};
pub use x86::{X86InlineAsmReg, X86InlineAsmRegClass};

#[derive(Copy, Clone, Encodable, Decodable, Debug, Eq, PartialEq, Hash)]
pub enum InlineAsmArch {
    X86,
    X86_64,
    Arm,
    AArch64,
    RiscV32,
    RiscV64,
    Nvptx64,
    Hexagon,
    Mips,
    Mips64,
    PowerPC,
    PowerPC64,
    S390x,
    SpirV,
    Wasm32,
    Wasm64,
    Bpf,
    Avr,
    Msp430,
}

impl FromStr for InlineAsmArch {
    type Err = ();

    fn from_str(s: &str) -> Result<InlineAsmArch, ()> {
        match s {
            "x86" => Ok(Self::X86),
            "x86_64" => Ok(Self::X86_64),
            "arm" => Ok(Self::Arm),
            "aarch64" => Ok(Self::AArch64),
            "riscv32" => Ok(Self::RiscV32),
            "riscv64" => Ok(Self::RiscV64),
            "nvptx64" => Ok(Self::Nvptx64),
            "powerpc" => Ok(Self::PowerPC),
            "powerpc64" => Ok(Self::PowerPC64),
            "hexagon" => Ok(Self::Hexagon),
            "mips" => Ok(Self::Mips),
            "mips64" => Ok(Self::Mips64),
            "s390x" => Ok(Self::S390x),
            "spirv" => Ok(Self::SpirV),
            "wasm32" => Ok(Self::Wasm32),
            "wasm64" => Ok(Self::Wasm64),
            "bpf" => Ok(Self::Bpf),
            "avr" => Ok(Self::Avr),
            "msp430" => Ok(Self::Msp430),
            _ => Err(()),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub enum InlineAsmReg {
    X86(X86InlineAsmReg),
    Arm(ArmInlineAsmReg),
    AArch64(AArch64InlineAsmReg),
    RiscV(RiscVInlineAsmReg),
    Nvptx(NvptxInlineAsmReg),
    PowerPC(PowerPCInlineAsmReg),
    Hexagon(HexagonInlineAsmReg),
    Mips(MipsInlineAsmReg),
    S390x(S390xInlineAsmReg),
    SpirV(SpirVInlineAsmReg),
    Wasm(WasmInlineAsmReg),
    Bpf(BpfInlineAsmReg),
    Avr(AvrInlineAsmReg),
    Msp430(Msp430InlineAsmReg),
    // Placeholder for invalid register constraints for the current target
    Err,
}

impl InlineAsmReg {
    pub fn name(self) -> &'static str {
        match self {
            Self::X86(r) => r.name(),
            Self::Arm(r) => r.name(),
            Self::AArch64(r) => r.name(),
            Self::RiscV(r) => r.name(),
            Self::PowerPC(r) => r.name(),
            Self::Hexagon(r) => r.name(),
            Self::Mips(r) => r.name(),
            Self::S390x(r) => r.name(),
            Self::Bpf(r) => r.name(),
            Self::Avr(r) => r.name(),
            Self::Msp430(r) => r.name(),
            Self::Err => "<reg>",
        }
    }

    pub fn reg_class(self) -> InlineAsmRegClass {
        match self {
            Self::X86(r) => InlineAsmRegClass::X86(r.reg_class()),
            Self::Arm(r) => InlineAsmRegClass::Arm(r.reg_class()),
            Self::AArch64(r) => InlineAsmRegClass::AArch64(r.reg_class()),
            Self::RiscV(r) => InlineAsmRegClass::RiscV(r.reg_class()),
            Self::PowerPC(r) => InlineAsmRegClass::PowerPC(r.reg_class()),
            Self::Hexagon(r) => InlineAsmRegClass::Hexagon(r.reg_class()),
            Self::Mips(r) => InlineAsmRegClass::Mips(r.reg_class()),
            Self::S390x(r) => InlineAsmRegClass::S390x(r.reg_class()),
            Self::Bpf(r) => InlineAsmRegClass::Bpf(r.reg_class()),
            Self::Avr(r) => InlineAsmRegClass::Avr(r.reg_class()),
            Self::Msp430(r) => InlineAsmRegClass::Msp430(r.reg_class()),
            Self::Err => InlineAsmRegClass::Err,
        }
    }

    pub fn parse(arch: InlineAsmArch, name: Symbol) -> Result<Self, &'static str> {
        // FIXME: use direct symbol comparison for register names
        // Use `Symbol::as_str` instead of `Symbol::with` here because `has_feature` may access `Symbol`.
        let name = name.as_str();
        Ok(match arch {
            InlineAsmArch::X86 | InlineAsmArch::X86_64 => Self::X86(X86InlineAsmReg::parse(name)?),
            InlineAsmArch::Arm => Self::Arm(ArmInlineAsmReg::parse(name)?),
            InlineAsmArch::AArch64 => Self::AArch64(AArch64InlineAsmReg::parse(name)?),
            InlineAsmArch::RiscV32 | InlineAsmArch::RiscV64 => {
                Self::RiscV(RiscVInlineAsmReg::parse(name)?)
            }
            InlineAsmArch::Nvptx64 => Self::Nvptx(NvptxInlineAsmReg::parse(name)?),
            InlineAsmArch::PowerPC | InlineAsmArch::PowerPC64 => {
                Self::PowerPC(PowerPCInlineAsmReg::parse(name)?)
            }
            InlineAsmArch::Hexagon => Self::Hexagon(HexagonInlineAsmReg::parse(name)?),
            InlineAsmArch::Mips | InlineAsmArch::Mips64 => {
                Self::Mips(MipsInlineAsmReg::parse(name)?)
            }
            InlineAsmArch::S390x => Self::S390x(S390xInlineAsmReg::parse(name)?),
            InlineAsmArch::SpirV => Self::SpirV(SpirVInlineAsmReg::parse(name)?),
            InlineAsmArch::Wasm32 | InlineAsmArch::Wasm64 => {
                Self::Wasm(WasmInlineAsmReg::parse(name)?)
            }
            InlineAsmArch::Bpf => Self::Bpf(BpfInlineAsmReg::parse(name)?),
            InlineAsmArch::Avr => Self::Avr(AvrInlineAsmReg::parse(name)?),
            InlineAsmArch::Msp430 => Self::Msp430(Msp430InlineAsmReg::parse(name)?),
        })
    }

    pub fn validate(
        self,
        arch: InlineAsmArch,
        reloc_model: RelocModel,
        target_features: &FxIndexSet<Symbol>,
        target: &Target,
        is_clobber: bool,
    ) -> Result<(), &'static str> {
        match self {
            Self::X86(r) => r.validate(arch, reloc_model, target_features, target, is_clobber),
            Self::Arm(r) => r.validate(arch, reloc_model, target_features, target, is_clobber),
            Self::AArch64(r) => r.validate(arch, reloc_model, target_features, target, is_clobber),
            Self::RiscV(r) => r.validate(arch, reloc_model, target_features, target, is_clobber),
            Self::PowerPC(r) => r.validate(arch, reloc_model, target_features, target, is_clobber),
            Self::Hexagon(r) => r.validate(arch, reloc_model, target_features, target, is_clobber),
            Self::Mips(r) => r.validate(arch, reloc_model, target_features, target, is_clobber),
            Self::S390x(r) => r.validate(arch, reloc_model, target_features, target, is_clobber),
            Self::Bpf(r) => r.validate(arch, reloc_model, target_features, target, is_clobber),
            Self::Avr(r) => r.validate(arch, reloc_model, target_features, target, is_clobber),
            Self::Msp430(r) => r.validate(arch, reloc_model, target_features, target, is_clobber),
            Self::Err => unreachable!(),
        }
    }

    // NOTE: This function isn't used at the moment, but is needed to support
    // falling back to an external assembler.
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        arch: InlineAsmArch,
        modifier: Option<char>,
    ) -> fmt::Result {
        match self {
            Self::X86(r) => r.emit(out, arch, modifier),
            Self::Arm(r) => r.emit(out, arch, modifier),
            Self::AArch64(r) => r.emit(out, arch, modifier),
            Self::RiscV(r) => r.emit(out, arch, modifier),
            Self::PowerPC(r) => r.emit(out, arch, modifier),
            Self::Hexagon(r) => r.emit(out, arch, modifier),
            Self::Mips(r) => r.emit(out, arch, modifier),
            Self::S390x(r) => r.emit(out, arch, modifier),
            Self::Bpf(r) => r.emit(out, arch, modifier),
            Self::Avr(r) => r.emit(out, arch, modifier),
            Self::Msp430(r) => r.emit(out, arch, modifier),
            Self::Err => unreachable!("Use of InlineAsmReg::Err"),
        }
    }

    pub fn overlapping_regs(self, mut cb: impl FnMut(InlineAsmReg)) {
        match self {
            Self::X86(r) => r.overlapping_regs(|r| cb(Self::X86(r))),
            Self::Arm(r) => r.overlapping_regs(|r| cb(Self::Arm(r))),
            Self::AArch64(_) => cb(self),
            Self::RiscV(_) => cb(self),
            Self::PowerPC(r) => r.overlapping_regs(|r| cb(Self::PowerPC(r))),
            Self::Hexagon(r) => r.overlapping_regs(|r| cb(Self::Hexagon(r))),
            Self::Mips(_) => cb(self),
            Self::S390x(_) => cb(self),
            Self::Bpf(r) => r.overlapping_regs(|r| cb(Self::Bpf(r))),
            Self::Avr(r) => r.overlapping_regs(|r| cb(Self::Avr(r))),
            Self::Msp430(_) => cb(self),
            Self::Err => unreachable!("Use of InlineAsmReg::Err"),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub enum InlineAsmRegClass {
    X86(X86InlineAsmRegClass),
    Arm(ArmInlineAsmRegClass),
    AArch64(AArch64InlineAsmRegClass),
    RiscV(RiscVInlineAsmRegClass),
    Nvptx(NvptxInlineAsmRegClass),
    PowerPC(PowerPCInlineAsmRegClass),
    Hexagon(HexagonInlineAsmRegClass),
    Mips(MipsInlineAsmRegClass),
    S390x(S390xInlineAsmRegClass),
    SpirV(SpirVInlineAsmRegClass),
    Wasm(WasmInlineAsmRegClass),
    Bpf(BpfInlineAsmRegClass),
    Avr(AvrInlineAsmRegClass),
    Msp430(Msp430InlineAsmRegClass),
    // Placeholder for invalid register constraints for the current target
    Err,
}

impl InlineAsmRegClass {
    pub fn name(self) -> Symbol {
        match self {
            Self::X86(r) => r.name(),
            Self::Arm(r) => r.name(),
            Self::AArch64(r) => r.name(),
            Self::RiscV(r) => r.name(),
            Self::Nvptx(r) => r.name(),
            Self::PowerPC(r) => r.name(),
            Self::Hexagon(r) => r.name(),
            Self::Mips(r) => r.name(),
            Self::S390x(r) => r.name(),
            Self::SpirV(r) => r.name(),
            Self::Wasm(r) => r.name(),
            Self::Bpf(r) => r.name(),
            Self::Avr(r) => r.name(),
            Self::Msp430(r) => r.name(),
            Self::Err => rustc_span::symbol::sym::reg,
        }
    }

    /// Returns a suggested register class to use for this type. This is called
    /// when `supported_types` fails to give a better error
    /// message to the user.
    pub fn suggest_class(self, arch: InlineAsmArch, ty: InlineAsmType) -> Option<Self> {
        match self {
            Self::X86(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::X86),
            Self::Arm(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::Arm),
            Self::AArch64(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::AArch64),
            Self::RiscV(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::RiscV),
            Self::Nvptx(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::Nvptx),
            Self::PowerPC(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::PowerPC),
            Self::Hexagon(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::Hexagon),
            Self::Mips(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::Mips),
            Self::S390x(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::S390x),
            Self::SpirV(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::SpirV),
            Self::Wasm(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::Wasm),
            Self::Bpf(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::Bpf),
            Self::Avr(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::Avr),
            Self::Msp430(r) => r.suggest_class(arch, ty).map(InlineAsmRegClass::Msp430),
            Self::Err => unreachable!("Use of InlineAsmRegClass::Err"),
        }
    }

    /// Returns a suggested template modifier to use for this type and an
    /// example of a register named formatted with it.
    ///
    /// Such suggestions are useful if a type smaller than the full register
    /// size is used and a modifier can be used to point to the subregister of
    /// the correct size.
    pub fn suggest_modifier(
        self,
        arch: InlineAsmArch,
        ty: InlineAsmType,
    ) -> Option<(char, &'static str)> {
        match self {
            Self::X86(r) => r.suggest_modifier(arch, ty),
            Self::Arm(r) => r.suggest_modifier(arch, ty),
            Self::AArch64(r) => r.suggest_modifier(arch, ty),
            Self::RiscV(r) => r.suggest_modifier(arch, ty),
            Self::Nvptx(r) => r.suggest_modifier(arch, ty),
            Self::PowerPC(r) => r.suggest_modifier(arch, ty),
            Self::Hexagon(r) => r.suggest_modifier(arch, ty),
            Self::Mips(r) => r.suggest_modifier(arch, ty),
            Self::S390x(r) => r.suggest_modifier(arch, ty),
            Self::SpirV(r) => r.suggest_modifier(arch, ty),
            Self::Wasm(r) => r.suggest_modifier(arch, ty),
            Self::Bpf(r) => r.suggest_modifier(arch, ty),
            Self::Avr(r) => r.suggest_modifier(arch, ty),
            Self::Msp430(r) => r.suggest_modifier(arch, ty),
            Self::Err => unreachable!("Use of InlineAsmRegClass::Err"),
        }
    }

    /// Returns the default modifier for this register and an example of a
    /// register named formatted with it.
    ///
    /// This is only needed when the register class can suggest a modifier, so
    /// that the user can be shown how to get the default behavior without a
    /// warning.
    pub fn default_modifier(self, arch: InlineAsmArch) -> Option<(char, &'static str)> {
        match self {
            Self::X86(r) => r.default_modifier(arch),
            Self::Arm(r) => r.default_modifier(arch),
            Self::AArch64(r) => r.default_modifier(arch),
            Self::RiscV(r) => r.default_modifier(arch),
            Self::Nvptx(r) => r.default_modifier(arch),
            Self::PowerPC(r) => r.default_modifier(arch),
            Self::Hexagon(r) => r.default_modifier(arch),
            Self::Mips(r) => r.default_modifier(arch),
            Self::S390x(r) => r.default_modifier(arch),
            Self::SpirV(r) => r.default_modifier(arch),
            Self::Wasm(r) => r.default_modifier(arch),
            Self::Bpf(r) => r.default_modifier(arch),
            Self::Avr(r) => r.default_modifier(arch),
            Self::Msp430(r) => r.default_modifier(arch),
            Self::Err => unreachable!("Use of InlineAsmRegClass::Err"),
        }
    }

    /// Returns a list of supported types for this register class, each with an
    /// options target feature required to use this type.
    pub fn supported_types(
        self,
        arch: InlineAsmArch,
    ) -> &'static [(InlineAsmType, Option<Symbol>)] {
        match self {
            Self::X86(r) => r.supported_types(arch),
            Self::Arm(r) => r.supported_types(arch),
            Self::AArch64(r) => r.supported_types(arch),
            Self::RiscV(r) => r.supported_types(arch),
            Self::Nvptx(r) => r.supported_types(arch),
            Self::PowerPC(r) => r.supported_types(arch),
            Self::Hexagon(r) => r.supported_types(arch),
            Self::Mips(r) => r.supported_types(arch),
            Self::S390x(r) => r.supported_types(arch),
            Self::SpirV(r) => r.supported_types(arch),
            Self::Wasm(r) => r.supported_types(arch),
            Self::Bpf(r) => r.supported_types(arch),
            Self::Avr(r) => r.supported_types(arch),
            Self::Msp430(r) => r.supported_types(arch),
            Self::Err => unreachable!("Use of InlineAsmRegClass::Err"),
        }
    }

    pub fn parse(arch: InlineAsmArch, name: Symbol) -> Result<Self, &'static str> {
        Ok(match arch {
            InlineAsmArch::X86 | InlineAsmArch::X86_64 => {
                Self::X86(X86InlineAsmRegClass::parse(name)?)
            }
            InlineAsmArch::Arm => Self::Arm(ArmInlineAsmRegClass::parse(name)?),
            InlineAsmArch::AArch64 => Self::AArch64(AArch64InlineAsmRegClass::parse(name)?),
            InlineAsmArch::RiscV32 | InlineAsmArch::RiscV64 => {
                Self::RiscV(RiscVInlineAsmRegClass::parse(name)?)
            }
            InlineAsmArch::Nvptx64 => Self::Nvptx(NvptxInlineAsmRegClass::parse(name)?),
            InlineAsmArch::PowerPC | InlineAsmArch::PowerPC64 => {
                Self::PowerPC(PowerPCInlineAsmRegClass::parse(name)?)
            }
            InlineAsmArch::Hexagon => Self::Hexagon(HexagonInlineAsmRegClass::parse(name)?),
            InlineAsmArch::Mips | InlineAsmArch::Mips64 => {
                Self::Mips(MipsInlineAsmRegClass::parse(name)?)
            }
            InlineAsmArch::S390x => Self::S390x(S390xInlineAsmRegClass::parse(name)?),
            InlineAsmArch::SpirV => Self::SpirV(SpirVInlineAsmRegClass::parse(name)?),
            InlineAsmArch::Wasm32 | InlineAsmArch::Wasm64 => {
                Self::Wasm(WasmInlineAsmRegClass::parse(name)?)
            }
            InlineAsmArch::Bpf => Self::Bpf(BpfInlineAsmRegClass::parse(name)?),
            InlineAsmArch::Avr => Self::Avr(AvrInlineAsmRegClass::parse(name)?),
            InlineAsmArch::Msp430 => Self::Msp430(Msp430InlineAsmRegClass::parse(name)?),
        })
    }

    /// Returns the list of template modifiers that can be used with this
    /// register class.
    pub fn valid_modifiers(self, arch: InlineAsmArch) -> &'static [char] {
        match self {
            Self::X86(r) => r.valid_modifiers(arch),
            Self::Arm(r) => r.valid_modifiers(arch),
            Self::AArch64(r) => r.valid_modifiers(arch),
            Self::RiscV(r) => r.valid_modifiers(arch),
            Self::Nvptx(r) => r.valid_modifiers(arch),
            Self::PowerPC(r) => r.valid_modifiers(arch),
            Self::Hexagon(r) => r.valid_modifiers(arch),
            Self::Mips(r) => r.valid_modifiers(arch),
            Self::S390x(r) => r.valid_modifiers(arch),
            Self::SpirV(r) => r.valid_modifiers(arch),
            Self::Wasm(r) => r.valid_modifiers(arch),
            Self::Bpf(r) => r.valid_modifiers(arch),
            Self::Avr(r) => r.valid_modifiers(arch),
            Self::Msp430(r) => r.valid_modifiers(arch),
            Self::Err => unreachable!("Use of InlineAsmRegClass::Err"),
        }
    }

    /// Returns whether registers in this class can only be used as clobbers
    /// and not as inputs/outputs.
    pub fn is_clobber_only(self, arch: InlineAsmArch) -> bool {
        self.supported_types(arch).is_empty()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub enum InlineAsmRegOrRegClass {
    Reg(InlineAsmReg),
    RegClass(InlineAsmRegClass),
}

impl InlineAsmRegOrRegClass {
    pub fn reg_class(self) -> InlineAsmRegClass {
        match self {
            Self::Reg(r) => r.reg_class(),
            Self::RegClass(r) => r,
        }
    }
}

impl fmt::Display for InlineAsmRegOrRegClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reg(r) => write!(f, "\"{}\"", r.name()),
            Self::RegClass(r) => write!(f, "{}", r.name()),
        }
    }
}

/// Set of types which can be used with a particular register class.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum InlineAsmType {
    I8,
    I16,
    I32,
    I64,
    I128,
    F32,
    F64,
    VecI8(u64),
    VecI16(u64),
    VecI32(u64),
    VecI64(u64),
    VecI128(u64),
    VecF32(u64),
    VecF64(u64),
}

impl InlineAsmType {
    pub fn is_integer(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::I128)
    }

    pub fn size(self) -> Size {
        Size::from_bytes(match self {
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::I128 => 16,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::VecI8(n) => n * 1,
            Self::VecI16(n) => n * 2,
            Self::VecI32(n) => n * 4,
            Self::VecI64(n) => n * 8,
            Self::VecI128(n) => n * 16,
            Self::VecF32(n) => n * 4,
            Self::VecF64(n) => n * 8,
        })
    }
}

impl fmt::Display for InlineAsmType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::I8 => f.write_str("i8"),
            Self::I16 => f.write_str("i16"),
            Self::I32 => f.write_str("i32"),
            Self::I64 => f.write_str("i64"),
            Self::I128 => f.write_str("i128"),
            Self::F32 => f.write_str("f32"),
            Self::F64 => f.write_str("f64"),
            Self::VecI8(n) => write!(f, "i8x{n}"),
            Self::VecI16(n) => write!(f, "i16x{n}"),
            Self::VecI32(n) => write!(f, "i32x{n}"),
            Self::VecI64(n) => write!(f, "i64x{n}"),
            Self::VecI128(n) => write!(f, "i128x{n}"),
            Self::VecF32(n) => write!(f, "f32x{n}"),
            Self::VecF64(n) => write!(f, "f64x{n}"),
        }
    }
}

/// Returns the full set of allocatable registers for a given architecture.
///
/// The registers are structured as a map containing the set of allocatable
/// registers in each register class. A particular register may be allocatable
/// from multiple register classes, in which case it will appear multiple times
/// in the map.
// NOTE: This function isn't used at the moment, but is needed to support
// falling back to an external assembler.
pub fn allocatable_registers(
    arch: InlineAsmArch,
    reloc_model: RelocModel,
    target_features: &FxIndexSet<Symbol>,
    target: &crate::spec::Target,
) -> FxHashMap<InlineAsmRegClass, FxIndexSet<InlineAsmReg>> {
    match arch {
        InlineAsmArch::X86 | InlineAsmArch::X86_64 => {
            let mut map = x86::regclass_map();
            x86::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
        InlineAsmArch::Arm => {
            let mut map = arm::regclass_map();
            arm::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
        InlineAsmArch::AArch64 => {
            let mut map = aarch64::regclass_map();
            aarch64::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
        InlineAsmArch::RiscV32 | InlineAsmArch::RiscV64 => {
            let mut map = riscv::regclass_map();
            riscv::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
        InlineAsmArch::Nvptx64 => {
            let mut map = nvptx::regclass_map();
            nvptx::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
        InlineAsmArch::PowerPC | InlineAsmArch::PowerPC64 => {
            let mut map = powerpc::regclass_map();
            powerpc::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
        InlineAsmArch::Hexagon => {
            let mut map = hexagon::regclass_map();
            hexagon::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
        InlineAsmArch::Mips | InlineAsmArch::Mips64 => {
            let mut map = mips::regclass_map();
            mips::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
        InlineAsmArch::S390x => {
            let mut map = s390x::regclass_map();
            s390x::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
        InlineAsmArch::SpirV => {
            let mut map = spirv::regclass_map();
            spirv::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
        InlineAsmArch::Wasm32 | InlineAsmArch::Wasm64 => {
            let mut map = wasm::regclass_map();
            wasm::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
        InlineAsmArch::Bpf => {
            let mut map = bpf::regclass_map();
            bpf::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
        InlineAsmArch::Avr => {
            let mut map = avr::regclass_map();
            avr::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
        InlineAsmArch::Msp430 => {
            let mut map = msp430::regclass_map();
            msp430::fill_reg_map(arch, reloc_model, target_features, target, &mut map);
            map
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub enum InlineAsmClobberAbi {
    X86,
    X86_64Win,
    X86_64SysV,
    Arm,
    AArch64,
    AArch64NoX18,
    RiscV,
}

impl InlineAsmClobberAbi {
    /// Parses a clobber ABI for the given target, or returns a list of supported
    /// clobber ABIs for the target.
    pub fn parse(
        arch: InlineAsmArch,
        target: &Target,
        name: Symbol,
    ) -> Result<Self, &'static [&'static str]> {
        let name = name.as_str();
        match arch {
            InlineAsmArch::X86 => match name {
                "C" | "system" | "efiapi" | "cdecl" | "stdcall" | "fastcall" => {
                    Ok(InlineAsmClobberAbi::X86)
                }
                _ => Err(&["C", "system", "efiapi", "cdecl", "stdcall", "fastcall"]),
            },
            InlineAsmArch::X86_64 => match name {
                "C" | "system" if !target.is_like_windows => Ok(InlineAsmClobberAbi::X86_64SysV),
                "C" | "system" if target.is_like_windows => Ok(InlineAsmClobberAbi::X86_64Win),
                "win64" | "efiapi" => Ok(InlineAsmClobberAbi::X86_64Win),
                "sysv64" => Ok(InlineAsmClobberAbi::X86_64SysV),
                _ => Err(&["C", "system", "efiapi", "win64", "sysv64"]),
            },
            InlineAsmArch::Arm => match name {
                "C" | "system" | "efiapi" | "aapcs" => Ok(InlineAsmClobberAbi::Arm),
                _ => Err(&["C", "system", "efiapi", "aapcs"]),
            },
            InlineAsmArch::AArch64 => match name {
                "C" | "system" | "efiapi" => Ok(if aarch64::target_reserves_x18(target) {
                    InlineAsmClobberAbi::AArch64NoX18
                } else {
                    InlineAsmClobberAbi::AArch64
                }),
                _ => Err(&["C", "system", "efiapi"]),
            },
            InlineAsmArch::RiscV32 | InlineAsmArch::RiscV64 => match name {
                "C" | "system" | "efiapi" => Ok(InlineAsmClobberAbi::RiscV),
                _ => Err(&["C", "system", "efiapi"]),
            },
            _ => Err(&[]),
        }
    }

    /// Returns the set of registers which are clobbered by this ABI.
    pub fn clobbered_regs(self) -> &'static [InlineAsmReg] {
        macro_rules! clobbered_regs {
            ($arch:ident $arch_reg:ident {
                $(
                    $reg:ident,
                )*
            }) => {
                &[
                    $(InlineAsmReg::$arch($arch_reg::$reg),)*
                ]
            };
        }
        match self {
            InlineAsmClobberAbi::X86 => clobbered_regs! {
                X86 X86InlineAsmReg {
                    ax, cx, dx,

                    xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7,

                    k0, k1, k2, k3, k4, k5, k6, k7,

                    mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7,
                    st0, st1, st2, st3, st4, st5, st6, st7,
                }
            },
            InlineAsmClobberAbi::X86_64SysV => clobbered_regs! {
                X86 X86InlineAsmReg {
                    ax, cx, dx, si, di, r8, r9, r10, r11,

                    xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7,
                    xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15,
                    zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23,
                    zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31,

                    k0, k1, k2, k3, k4, k5, k6, k7,

                    mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7,
                    st0, st1, st2, st3, st4, st5, st6, st7,
                    tmm0, tmm1, tmm2, tmm3, tmm4, tmm5, tmm6, tmm7,
                }
            },
            InlineAsmClobberAbi::X86_64Win => clobbered_regs! {
                X86 X86InlineAsmReg {
                    // rdi and rsi are callee-saved on windows
                    ax, cx, dx, r8, r9, r10, r11,

                    // xmm6-xmm15 are callee-saved on windows, but we need to
                    // mark them as clobbered anyways because the upper portions
                    // of ymm6-ymm15 are volatile.
                    xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7,
                    xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15,
                    zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23,
                    zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31,

                    k0, k1, k2, k3, k4, k5, k6, k7,

                    mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7,
                    st0, st1, st2, st3, st4, st5, st6, st7,
                    tmm0, tmm1, tmm2, tmm3, tmm4, tmm5, tmm6, tmm7,
                }
            },
            InlineAsmClobberAbi::AArch64 => clobbered_regs! {
                AArch64 AArch64InlineAsmReg {
                    x0, x1, x2, x3, x4, x5, x6, x7,
                    x8, x9, x10, x11, x12, x13, x14, x15,
                    x16, x17, x18, x30,

                    // Technically the low 64 bits of v8-v15 are preserved, but
                    // we have no way of expressing this using clobbers.
                    v0, v1, v2, v3, v4, v5, v6, v7,
                    v8, v9, v10, v11, v12, v13, v14, v15,
                    v16, v17, v18, v19, v20, v21, v22, v23,
                    v24, v25, v26, v27, v28, v29, v30, v31,

                    p0, p1, p2, p3, p4, p5, p6, p7,
                    p8, p9, p10, p11, p12, p13, p14, p15,
                    ffr,

                }
            },
            InlineAsmClobberAbi::AArch64NoX18 => clobbered_regs! {
                AArch64 AArch64InlineAsmReg {
                    x0, x1, x2, x3, x4, x5, x6, x7,
                    x8, x9, x10, x11, x12, x13, x14, x15,
                    x16, x17, x30,

                    // Technically the low 64 bits of v8-v15 are preserved, but
                    // we have no way of expressing this using clobbers.
                    v0, v1, v2, v3, v4, v5, v6, v7,
                    v8, v9, v10, v11, v12, v13, v14, v15,
                    v16, v17, v18, v19, v20, v21, v22, v23,
                    v24, v25, v26, v27, v28, v29, v30, v31,

                    p0, p1, p2, p3, p4, p5, p6, p7,
                    p8, p9, p10, p11, p12, p13, p14, p15,
                    ffr,

                }
            },
            InlineAsmClobberAbi::Arm => clobbered_regs! {
                Arm ArmInlineAsmReg {
                    // r9 is either platform-reserved or callee-saved. Either
                    // way we don't need to clobber it.
                    r0, r1, r2, r3, r12, r14,

                    // The finest-grained register variant is used here so that
                    // partial uses of larger registers are properly handled.
                    s0, s1, s2, s3, s4, s5, s6, s7,
                    s8, s9, s10, s11, s12, s13, s14, s15,
                    // s16-s31 are callee-saved
                    d16, d17, d18, d19, d20, d21, d22, d23,
                    d24, d25, d26, d27, d28, d29, d30, d31,
                }
            },
            InlineAsmClobberAbi::RiscV => clobbered_regs! {
                RiscV RiscVInlineAsmReg {
                    // ra
                    x1,
                    // t0-t2
                    x5, x6, x7,
                    // a0-a7
                    x10, x11, x12, x13, x14, x15, x16, x17,
                    // t3-t6
                    x28, x29, x30, x31,
                    // ft0-ft7
                    f0, f1, f2, f3, f4, f5, f6, f7,
                    // fa0-fa7
                    f10, f11, f12, f13, f14, f15, f16, f17,
                    // ft8-ft11
                    f28, f29, f30, f31,

                    v0, v1, v2, v3, v4, v5, v6, v7,
                    v8, v9, v10, v11, v12, v13, v14, v15,
                    v16, v17, v18, v19, v20, v21, v22, v23,
                    v24, v25, v26, v27, v28, v29, v30, v31,
                }
            },
        }
    }
}
