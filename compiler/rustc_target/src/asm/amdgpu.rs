use std::fmt;

use rustc_span::Symbol;

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};

// Types are listed as SGPR_*/VGPR_* in llvm/lib/Target/AMDGPU/SIRegisterInfo.td
def_reg_class! {
    Amdgpu AmdgpuInlineAsmRegClass {
        sgpr32,
        sgpr64,
        sgpr96,
        sgpr128,
        sgpr256,
        sgpr512,
        vgpr16,
        vgpr32,
        vgpr64,
        vgpr96,
        vgpr128,
        vgpr160,
        vgpr192,
        vgpr224,
        vgpr256,
        vgpr288,
        vgpr320,
        vgpr352,
        vgpr384,
        vgpr512,
        vgpr1024,
    }
}

#[derive(
    Copy,
    Clone,
    rustc_macros::Encodable,
    rustc_macros::Decodable,
    Debug,
    Eq,
    PartialEq,
    PartialOrd,
    Hash,
    rustc_macros::HashStable_Generic
)]
pub enum AmdgpuInlineAsmRegClassType {
    Sgpr,
    Vgpr,
}

// See https://llvm.org/docs/AMDGPUOperandSyntax.html
impl AmdgpuInlineAsmRegClass {
    pub fn get_type(self) -> AmdgpuInlineAsmRegClassType {
        match self {
            Self::sgpr32
            | Self::sgpr64
            | Self::sgpr96
            | Self::sgpr128
            | Self::sgpr256
            | Self::sgpr512 => AmdgpuInlineAsmRegClassType::Sgpr,
            Self::vgpr16
            | Self::vgpr32
            | Self::vgpr64
            | Self::vgpr96
            | Self::vgpr128
            | Self::vgpr160
            | Self::vgpr192
            | Self::vgpr224
            | Self::vgpr256
            | Self::vgpr288
            | Self::vgpr320
            | Self::vgpr352
            | Self::vgpr384
            | Self::vgpr512
            | Self::vgpr1024 => AmdgpuInlineAsmRegClassType::Vgpr,
        }
    }

    /// Return size of the register class in bytes
    pub fn bytes(self) -> u32 {
        match self {
            Self::vgpr16 => 16 / 8,
            Self::sgpr32 | Self::vgpr32 => 32 / 8,
            Self::sgpr64 | Self::vgpr64 => 64 / 8,
            Self::sgpr96 | Self::vgpr96 => 96 / 8,
            Self::sgpr128 | Self::vgpr128 => 128 / 8,
            Self::vgpr160 => 160 / 8,
            Self::vgpr192 => 192 / 8,
            Self::vgpr224 => 224 / 8,
            Self::sgpr256 | Self::vgpr256 => 256 / 8,
            Self::vgpr288 => 288 / 8,
            Self::vgpr320 => 320 / 8,
            Self::vgpr352 => 352 / 8,
            Self::vgpr384 => 384 / 8,
            Self::sgpr512 | Self::vgpr512 => 512 / 8,
            Self::vgpr1024 => 1024 / 8,
        }
    }

    fn from_type(ty: AmdgpuInlineAsmRegClassType, bytes: u32) -> Option<Self> {
        let class = match ty {
            AmdgpuInlineAsmRegClassType::Sgpr => match bytes * 8 {
                32 => Self::sgpr32,
                64 => Self::sgpr64,
                96 => Self::sgpr96,
                128 => Self::sgpr128,
                256 => Self::sgpr256,
                512 => Self::sgpr512,
                _ => return None,
            },
            AmdgpuInlineAsmRegClassType::Vgpr => match bytes * 8 {
                16 => Self::vgpr16,
                32 => Self::vgpr32,
                64 => Self::vgpr64,
                96 => Self::vgpr96,
                128 => Self::vgpr128,
                160 => Self::vgpr160,
                192 => Self::vgpr192,
                224 => Self::vgpr224,
                256 => Self::vgpr256,
                288 => Self::vgpr288,
                320 => Self::vgpr320,
                352 => Self::vgpr352,
                384 => Self::vgpr384,
                512 => Self::vgpr512,
                1024 => Self::vgpr1024,
                _ => return None,
            },
        };
        Some(class)
    }

    pub fn valid_modifiers(self, _arch: InlineAsmArch) -> &'static [char] {
        &[]
    }

    pub fn suggest_class(self, _arch: InlineAsmArch, ty: InlineAsmType) -> Option<Self> {
        // Suggest VGPR for everything as VGPRs have more uses
        Some(match ty {
            InlineAsmType::I16 => Self::vgpr16,
            InlineAsmType::I32 => Self::vgpr32,
            InlineAsmType::I64 => Self::vgpr64,
            InlineAsmType::I128 => Self::vgpr128,
            InlineAsmType::F16 => Self::vgpr16,
            InlineAsmType::F32 => Self::vgpr32,
            InlineAsmType::F64 => Self::vgpr64,
            _ => {
                let bytes = match ty {
                    InlineAsmType::VecI16(n) => n * (16 / 8),
                    InlineAsmType::VecI32(n) => n * (32 / 8),
                    InlineAsmType::VecI64(n) => n * (64 / 8),
                    InlineAsmType::VecI128(n) => n * (128 / 8),
                    InlineAsmType::VecF16(n) => n * (16 / 8),
                    InlineAsmType::VecF32(n) => n * (32 / 8),
                    InlineAsmType::VecF64(n) => n * (64 / 8),
                    _ => return None,
                };
                return Self::from_type(AmdgpuInlineAsmRegClassType::Vgpr, bytes as u32);
            }
        })
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
            Self::vgpr16 => types! { _: I16, F16; },
            Self::sgpr32 | Self::vgpr32 => types! { _: I16, I32, F16, F32,
                VecI16(32 / 16),
                VecF16(32 / 16);
            },
            Self::sgpr64 | Self::vgpr64 => types! {
                _: I64, F64, VecI16(64 / 16), VecI32(64 / 32),
                VecF16(64 / 16), VecF32(64 / 32);
            },
            Self::sgpr96 | Self::vgpr96 => types! { _: VecI32(96 / 32), VecF32(96 / 32); },
            Self::sgpr128 | Self::vgpr128 => types! { _: I128,
                VecI16(128 / 16), VecI32(128 / 32), VecI64(128 / 64),
                VecF16(128 / 16), VecF32(128 / 32), VecF64(128 / 64);
            },
            Self::vgpr160 => types! { _: VecI32(160 / 32), VecF32(160 / 32); },
            Self::vgpr192 => types! { _:
                VecI32(192 / 32), VecI64(192 / 64),
                VecF32(192 / 32), VecF64(192 / 64);
            },
            Self::vgpr224 => types! { _: VecI32(224 / 32), VecF32(224 / 32); },
            Self::sgpr256 => types! { _:
                VecI16(256 / 16), VecI32(256 / 32), VecI64(256 / 64),
                VecF16(256 / 16), VecF32(256 / 32), VecF64(256 / 64);
            },
            Self::vgpr256 => types! { _:
                VecI16(256 / 16), VecI32(256 / 32),
                VecF16(256 / 16), VecF32(256 / 32), VecF64(256 / 64);
            },
            Self::vgpr288 => types! { _: VecI32(288 / 32), VecF32(288 / 32); },
            Self::vgpr320 => types! { _: VecI32(320 / 32), VecF32(320 / 32); },
            Self::vgpr352 => types! { _: VecI32(352 / 32), VecF32(352 / 32); },
            Self::vgpr384 => types! { _: VecI32(384 / 32), VecF32(384 / 32); },
            Self::sgpr512 => types! { _:
                VecI16(512 / 16), VecI32(512 / 32), VecI64(512 / 64),
                VecF16(512 / 16), VecF32(512 / 32), VecF64(512 / 64);
            },
            Self::vgpr512 => types! { _:
                VecI16(512 / 16), VecI32(512 / 32),
                VecF16(512 / 16), VecF32(512 / 32);
            },
            Self::vgpr1024 => types! { _: VecF32(1024 / 32); },
        }
    }

    /// The number of supported registers in this class.
    /// The returned number is the length, so supported register
    /// indices are 0 to max_num()-1.
    fn max_num(self) -> u32 {
        if self == AmdgpuInlineAsmRegClass::vgpr16 {
            return 512;
        }
        let size = self.bytes();
        match self.get_type() {
            AmdgpuInlineAsmRegClassType::Sgpr => 106 - (size / 4 - 1),
            AmdgpuInlineAsmRegClassType::Vgpr => 256 - (size / 4 - 1),
        }
    }

    /// Get register class from prefix.
    fn parse_prefix(prefix: char) -> Result<AmdgpuInlineAsmRegClassType, &'static str> {
        match prefix {
            's' => Ok(AmdgpuInlineAsmRegClassType::Sgpr),
            'v' => Ok(AmdgpuInlineAsmRegClassType::Vgpr),
            _ => Err("unknown register prefix"),
        }
    }
}

impl AmdgpuInlineAsmRegClassType {
    /// Prefix when printed and register constraint in LLVM.
    fn prefix(self) -> &'static str {
        match self {
            AmdgpuInlineAsmRegClassType::Sgpr => "s",
            AmdgpuInlineAsmRegClassType::Vgpr => "v",
        }
    }
}

#[derive(
    Copy,
    Clone,
    rustc_macros::Encodable,
    rustc_macros::Decodable,
    Debug,
    Eq,
    PartialEq,
    PartialOrd,
    Hash,
    rustc_macros::HashStable_Generic
)]
enum AmdgpuRegRange {
    /// Low 16-bit of a register
    Low(u32),
    /// High 16-bit of a register
    High(u32),
    /// One or more 32-bit registers, in the inclusive range
    Range { start: u32, end: u32 },
}

#[derive(
    Copy,
    Clone,
    rustc_macros::Encodable,
    rustc_macros::Decodable,
    Debug,
    Eq,
    PartialEq,
    PartialOrd,
    Hash,
    rustc_macros::HashStable_Generic
)]
#[allow(non_camel_case_types)]
pub struct AmdgpuInlineAsmReg {
    class: AmdgpuInlineAsmRegClassType,
    range: AmdgpuRegRange,
}

impl AmdgpuInlineAsmReg {
    pub fn name(self) -> String {
        let c = self.class.prefix();
        match self.range {
            AmdgpuRegRange::Low(n) => format!("{c}{n}.l"),
            AmdgpuRegRange::High(n) => format!("{c}{n}.h"),
            AmdgpuRegRange::Range { start, end } if start == end => format!("{c}{start}"),
            AmdgpuRegRange::Range { start, end } => format!("{c}[{start}:{end}]"),
        }
    }

    /// Size of the register in bytes
    fn bytes(self) -> u32 {
        match self.range {
            AmdgpuRegRange::Low(_) | AmdgpuRegRange::High(_) => 2,
            AmdgpuRegRange::Range { start, end } => ((end - start) + 1) * 4,
        }
    }

    pub fn reg_class(self) -> AmdgpuInlineAsmRegClass {
        AmdgpuInlineAsmRegClass::from_type(self.class, self.bytes())
            .expect("Failed to emit invalid amdgpu register class")
    }

    pub fn parse(name: &str) -> Result<Self, &'static str> {
        if name.is_empty() {
            return Err("invalid empty register");
        }
        let class = AmdgpuInlineAsmRegClass::parse_prefix(name.chars().next().unwrap())?;
        // Form with range, e.g. s[2:3]
        let res;
        if name[1..].starts_with('[') {
            if !name.ends_with(']') {
                return Err("invalid register, missing closing bracket");
            }
            if let Some((start, end)) = name[2..name.len() - 1].split_once(':') {
                let Ok(start) = start.parse() else {
                    return Err("invalid register range start");
                };
                let Ok(end) = end.parse() else {
                    return Err("invalid register range end");
                };

                // Check range
                if start > end {
                    return Err("invalid reversed register range");
                }

                if let Some(class) =
                    AmdgpuInlineAsmRegClass::from_type(class, ((end - start) + 1) * 4)
                {
                    if end >= class.max_num() {
                        return Err("too large register for this class");
                    }
                } else {
                    return Err("invalid register size for this class");
                }
                res = Self { class, range: AmdgpuRegRange::Range { start, end } };
            } else {
                return Err("invalid register range");
            }
        } else {
            let parse_num = |core: &str| {
                let Ok(start) = core.parse() else {
                    return Err("invalid register number");
                };

                if let Some(class) = AmdgpuInlineAsmRegClass::from_type(class, 4) {
                    if start >= class.max_num() {
                        return Err("too large register for this class");
                    }
                } else {
                    return Err("invalid register size for this class");
                }

                Ok(start)
            };

            let name = &name[1..];
            let range = if let Some(name) = name.strip_suffix(".l") {
                if class == AmdgpuInlineAsmRegClassType::Sgpr {
                    return Err("invalid 16-bit SGPR register");
                }
                AmdgpuRegRange::Low(parse_num(name)?)
            } else if let Some(name) = name.strip_suffix(".h") {
                if class == AmdgpuInlineAsmRegClassType::Sgpr {
                    return Err("invalid 16-bit SGPR register");
                }
                AmdgpuRegRange::High(parse_num(name)?)
            } else {
                let start = parse_num(name)?;
                AmdgpuRegRange::Range { start, end: start }
            };
            res = Self { class, range };
        }
        Ok(res)
    }

    pub fn validate(
        self,
        _arch: super::InlineAsmArch,
        _reloc_model: crate::spec::RelocModel,
        _target_features: &rustc_data_structures::fx::FxIndexSet<Symbol>,
        _target: &crate::spec::Target,
        _is_clobber: bool,
    ) -> Result<(), &'static str> {
        Ok(())
    }
}

pub(super) fn fill_reg_map(
    _arch: super::InlineAsmArch,
    _reloc_model: crate::spec::RelocModel,
    _target_features: &rustc_data_structures::fx::FxIndexSet<Symbol>,
    _target: &crate::spec::Target,
    map: &mut rustc_data_structures::fx::FxHashMap<
        super::InlineAsmRegClass,
        rustc_data_structures::fx::FxIndexSet<super::InlineAsmReg>,
    >,
) {
    use super::{InlineAsmReg, InlineAsmRegClass};

    #[allow(rustc::potential_query_instability)]
    for class in regclass_map().keys() {
        let InlineAsmRegClass::Amdgpu(class) = *class else { unreachable!("Must be amdgpu class") };
        if let Some(set) = map.get_mut(&InlineAsmRegClass::Amdgpu(class)) {
            if class == AmdgpuInlineAsmRegClass::vgpr16 {
                for i in 0..(class.max_num() / 2) {
                    set.insert(InlineAsmReg::Amdgpu(AmdgpuInlineAsmReg {
                        class: AmdgpuInlineAsmRegClassType::Vgpr,
                        range: AmdgpuRegRange::Low(i),
                    }));
                    set.insert(InlineAsmReg::Amdgpu(AmdgpuInlineAsmReg {
                        class: AmdgpuInlineAsmRegClassType::Vgpr,
                        range: AmdgpuRegRange::High(i),
                    }));
                }
            } else {
                for i in 0..class.max_num() {
                    set.insert(InlineAsmReg::Amdgpu(AmdgpuInlineAsmReg {
                        class: class.get_type(),
                        range: AmdgpuRegRange::Range { start: i, end: i + class.bytes() / 4 },
                    }));
                }
            }
        }
    }
}

impl AmdgpuInlineAsmReg {
    pub fn emit(
        self,
        out: &mut dyn fmt::Write,
        _arch: InlineAsmArch,
        _modifier: Option<char>,
    ) -> fmt::Result {
        out.write_str(&self.name())
    }

    pub fn overlapping_regs(self, mut cb: impl FnMut(AmdgpuInlineAsmReg)) {
        if self.class != AmdgpuInlineAsmRegClassType::Sgpr {
            // Overlapping 16-bit registers (not supported for sgprs)
            if let AmdgpuRegRange::Range { start, end } = self.range {
                for i in start..=end {
                    cb(AmdgpuInlineAsmReg { class: self.class, range: AmdgpuRegRange::Low(i) });
                    cb(AmdgpuInlineAsmReg { class: self.class, range: AmdgpuRegRange::High(i) });
                }
            }
        }

        // Overlapping 32-bit registers, up to size 32
        for size in 1..=32 {
            let (start, end) = match self.range {
                AmdgpuRegRange::Low(start) | AmdgpuRegRange::High(start) => (start, start),
                AmdgpuRegRange::Range { start, end } => (start, end),
            };

            let size_range = size - 1;
            for overlap_start in (start - size_range)..=end {
                cb(AmdgpuInlineAsmReg {
                    class: self.class,
                    range: AmdgpuRegRange::Range {
                        start: overlap_start,
                        end: overlap_start + size_range,
                    },
                });
            }
        }
    }
}
