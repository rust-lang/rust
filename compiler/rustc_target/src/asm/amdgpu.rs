use std::fmt;

use rustc_span::Symbol;

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};

// Types are listed as SGPR_*/VGPR_* in llvm/lib/Target/AMDGPU/SIRegisterInfo.td

/// Amdgpu register classes
///
/// The number is the size of the register class in bits.
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
pub enum AmdgpuInlineAsmRegClass {
    Sgpr(u16),
    Vgpr(u16),
}

pub(super) fn regclass_map() -> rustc_data_structures::fx::FxHashMap<
    super::InlineAsmRegClass,
    rustc_data_structures::fx::FxIndexSet<super::InlineAsmReg>,
> {
    use rustc_data_structures::fx::{FxHashMap, FxIndexSet};

    use super::InlineAsmRegClass;
    let mut map = FxHashMap::default();

    // SGPR and VGPR sizes
    for i in [32, 64, 96, 128, 256, 512] {
        map.insert(
            InlineAsmRegClass::Amdgpu(AmdgpuInlineAsmRegClass::Sgpr(i)),
            FxIndexSet::default(),
        );
        map.insert(
            InlineAsmRegClass::Amdgpu(AmdgpuInlineAsmRegClass::Vgpr(i)),
            FxIndexSet::default(),
        );
    }

    // VGPR-only sizes
    for i in [16, 160, 192, 224, 288, 320, 352, 384, 1024] {
        map.insert(
            InlineAsmRegClass::Amdgpu(AmdgpuInlineAsmRegClass::Vgpr(i)),
            FxIndexSet::default(),
        );
    }

    map
}

// See https://llvm.org/docs/AMDGPUOperandSyntax.html
impl AmdgpuInlineAsmRegClass {
    /// Prefix when printed and register constraint in LLVM
    fn prefix(self) -> &'static str {
        match self {
            Self::Sgpr(_) => "s",
            Self::Vgpr(_) => "v",
        }
    }

    /// Return size of the register class in bytes
    fn bytes(self) -> u16 {
        let (Self::Sgpr(i) | Self::Vgpr(i)) = self;
        i / 8
    }

    /// Returns the name or `None` if this is not a valid register class
    fn try_get_name(self) -> Option<rustc_span::Symbol> {
        let s = match self {
            Self::Sgpr(32) => rustc_span::sym::sgpr32,
            Self::Sgpr(64) => rustc_span::sym::sgpr64,
            Self::Sgpr(96) => rustc_span::sym::sgpr96,
            Self::Sgpr(128) => rustc_span::sym::sgpr128,
            Self::Sgpr(256) => rustc_span::sym::sgpr256,
            Self::Sgpr(512) => rustc_span::sym::sgpr512,
            Self::Vgpr(16) => rustc_span::sym::vgpr16,
            Self::Vgpr(32) => rustc_span::sym::vgpr32,
            Self::Vgpr(64) => rustc_span::sym::vgpr64,
            Self::Vgpr(96) => rustc_span::sym::vgpr96,
            Self::Vgpr(128) => rustc_span::sym::vgpr128,
            Self::Vgpr(160) => rustc_span::sym::vgpr160,
            Self::Vgpr(192) => rustc_span::sym::vgpr192,
            Self::Vgpr(224) => rustc_span::sym::vgpr224,
            Self::Vgpr(256) => rustc_span::sym::vgpr256,
            Self::Vgpr(288) => rustc_span::sym::vgpr288,
            Self::Vgpr(320) => rustc_span::sym::vgpr320,
            Self::Vgpr(352) => rustc_span::sym::vgpr352,
            Self::Vgpr(384) => rustc_span::sym::vgpr384,
            Self::Vgpr(512) => rustc_span::sym::vgpr512,
            Self::Vgpr(1024) => rustc_span::sym::vgpr1024,
            _ => return None,
        };
        Some(s)
    }

    pub fn name(self) -> rustc_span::Symbol {
        self.try_get_name().expect("Invalid amdgpu register class")
    }

    pub fn parse(name: rustc_span::Symbol) -> Result<Self, &'static [rustc_span::Symbol]> {
        match name {
            rustc_span::sym::sgpr32 => Ok(Self::Sgpr(32)),
            rustc_span::sym::sgpr64 => Ok(Self::Sgpr(64)),
            rustc_span::sym::sgpr96 => Ok(Self::Sgpr(96)),
            rustc_span::sym::sgpr128 => Ok(Self::Sgpr(128)),
            rustc_span::sym::sgpr256 => Ok(Self::Sgpr(256)),
            rustc_span::sym::sgpr512 => Ok(Self::Sgpr(512)),
            rustc_span::sym::vgpr16 => Ok(Self::Vgpr(16)),
            rustc_span::sym::vgpr32 => Ok(Self::Vgpr(32)),
            rustc_span::sym::vgpr64 => Ok(Self::Vgpr(64)),
            rustc_span::sym::vgpr96 => Ok(Self::Vgpr(96)),
            rustc_span::sym::vgpr128 => Ok(Self::Vgpr(128)),
            rustc_span::sym::vgpr160 => Ok(Self::Vgpr(160)),
            rustc_span::sym::vgpr192 => Ok(Self::Vgpr(192)),
            rustc_span::sym::vgpr224 => Ok(Self::Vgpr(224)),
            rustc_span::sym::vgpr256 => Ok(Self::Vgpr(256)),
            rustc_span::sym::vgpr288 => Ok(Self::Vgpr(288)),
            rustc_span::sym::vgpr320 => Ok(Self::Vgpr(320)),
            rustc_span::sym::vgpr352 => Ok(Self::Vgpr(352)),
            rustc_span::sym::vgpr384 => Ok(Self::Vgpr(384)),
            rustc_span::sym::vgpr512 => Ok(Self::Vgpr(512)),
            rustc_span::sym::vgpr1024 => Ok(Self::Vgpr(1024)),
            _ => Err(&[
                rustc_span::sym::sgpr32,
                rustc_span::sym::sgpr64,
                rustc_span::sym::sgpr96,
                rustc_span::sym::sgpr128,
                rustc_span::sym::sgpr256,
                rustc_span::sym::sgpr512,
                rustc_span::sym::vgpr16,
                rustc_span::sym::vgpr32,
                rustc_span::sym::vgpr64,
                rustc_span::sym::vgpr96,
                rustc_span::sym::vgpr128,
                rustc_span::sym::vgpr160,
                rustc_span::sym::vgpr192,
                rustc_span::sym::vgpr224,
                rustc_span::sym::vgpr256,
                rustc_span::sym::vgpr288,
                rustc_span::sym::vgpr320,
                rustc_span::sym::vgpr352,
                rustc_span::sym::vgpr384,
                rustc_span::sym::vgpr512,
                rustc_span::sym::vgpr1024,
            ]),
        }
    }

    pub fn valid_modifiers(self, _arch: InlineAsmArch) -> &'static [char] {
        &[]
    }

    pub fn suggest_class(self, _arch: InlineAsmArch, ty: InlineAsmType) -> Option<Self> {
        // 8-bit types and f128 are not supported
        if matches!(
            ty,
            InlineAsmType::I8
                | InlineAsmType::VecI8(_)
                | InlineAsmType::F128
                | InlineAsmType::VecF128(_)
        ) {
            return None;
        }

        Some(Self::Vgpr(ty.size().bits().try_into().ok()?))
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
            Self::Vgpr(16) => types! { _: I16, F16; },
            Self::Sgpr(32) | Self::Vgpr(32) => types! { _: I16, I32, F16, F32,
                VecI16(32 / 16),
                VecF16(32 / 16);
            },
            Self::Sgpr(64) | Self::Vgpr(64) => types! {
                _: I64, F64, VecI16(64 / 16), VecI32(64 / 32),
                VecF16(64 / 16), VecF32(64 / 32);
            },
            Self::Sgpr(96) | Self::Vgpr(96) => types! { _: VecI32(96 / 32), VecF32(96 / 32); },
            Self::Sgpr(128) | Self::Vgpr(128) => types! { _: I128,
                VecI16(128 / 16), VecI32(128 / 32), VecI64(128 / 64),
                VecF16(128 / 16), VecF32(128 / 32), VecF64(128 / 64);
            },
            Self::Vgpr(160) => types! { _: VecI32(160 / 32), VecF32(160 / 32); },
            Self::Vgpr(192) => types! { _:
                VecI32(192 / 32), VecI64(192 / 64),
                VecF32(192 / 32), VecF64(192 / 64);
            },
            Self::Vgpr(224) => types! { _: VecI32(224 / 32), VecF32(224 / 32); },
            Self::Sgpr(256) => types! { _:
                VecI16(256 / 16), VecI32(256 / 32), VecI64(256 / 64),
                VecF16(256 / 16), VecF32(256 / 32), VecF64(256 / 64);
            },
            Self::Vgpr(256) => types! { _:
                VecI16(256 / 16), VecI32(256 / 32),
                VecF16(256 / 16), VecF32(256 / 32), VecF64(256 / 64);
            },
            Self::Vgpr(288) => types! { _: VecI32(288 / 32), VecF32(288 / 32); },
            Self::Vgpr(320) => types! { _: VecI32(320 / 32), VecF32(320 / 32); },
            Self::Vgpr(352) => types! { _: VecI32(352 / 32), VecF32(352 / 32); },
            Self::Vgpr(384) => types! { _: VecI32(384 / 32), VecF32(384 / 32); },
            Self::Sgpr(512) => types! { _:
                VecI16(512 / 16), VecI32(512 / 32), VecI64(512 / 64),
                VecF16(512 / 16), VecF32(512 / 32), VecF64(512 / 64);
            },
            Self::Vgpr(512) => types! { _:
                VecI16(512 / 16), VecI32(512 / 32),
                VecF16(512 / 16), VecF32(512 / 32);
            },
            Self::Vgpr(1024) => types! { _: VecF32(1024 / 32); },
            _ => panic!("Invalid amdgpu register class"),
        }
    }

    /// The number of supported registers in this class.
    /// The returned number is the length, so supported register
    /// indices are 0 to max_num()-1.
    fn max_num(self) -> u16 {
        if self == Self::Vgpr(16) {
            return 512;
        }
        let size = self.bytes();
        match self {
            Self::Sgpr(_) => 106 - (size / 4 - 1),
            Self::Vgpr(_) => 256 - (size / 4 - 1),
        }
    }

    /// Get register class from prefix and size.
    fn parse_with_prefix(prefix: char, bits: u16) -> Result<Self, &'static str> {
        let res = match prefix {
            's' => Self::Sgpr(bits),
            'v' => Self::Vgpr(bits),
            _ => return Err("unknown register prefix"),
        };

        // Check that the size is valid by converting it to a symbol
        if res.try_get_name().is_none() {
            return Err("invalid register size for this class");
        }

        Ok(res)
    }
}

/// Start index of a register.
///
/// Together with the register size this gives the range occupied by a register.
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
enum AmdgpuRegStart {
    /// Low 16-bit of the register at this index
    Low(u16),
    /// High 16-bit of the register at this index
    High(u16),
    /// One or more 32-bit registers, starting at this index
    Full(u16),
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
    class: AmdgpuInlineAsmRegClass,
    range: AmdgpuRegStart,
}

impl AmdgpuInlineAsmReg {
    pub fn name(self) -> String {
        let c = self.class.prefix();
        match self.range {
            AmdgpuRegStart::Low(n) => format!("{c}{n}.l"),
            AmdgpuRegStart::High(n) => format!("{c}{n}.h"),
            AmdgpuRegStart::Full(n) if self.class.bytes() == 4 => format!("{c}{n}"),
            AmdgpuRegStart::Full(n) => format!("{c}[{n}:{}]", n + self.class.bytes() / 4 - 1),
        }
    }

    pub fn reg_class(self) -> AmdgpuInlineAsmRegClass {
        self.class
    }

    pub fn parse(name: &str) -> Result<Self, &'static str> {
        if name.is_empty() {
            return Err("invalid empty register");
        }
        // s or v
        let prefix = name.chars().next().unwrap();
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

                let class =
                    AmdgpuInlineAsmRegClass::parse_with_prefix(prefix, ((end - start) + 1) * 32)?;
                if end >= class.max_num() {
                    return Err("too large register for this class");
                }
                res = Self { class, range: AmdgpuRegStart::Full(start) };
            } else {
                return Err("invalid register range");
            }
        } else {
            let parse_num = |core: &str| {
                let Ok(start) = core.parse() else {
                    return Err("invalid register number");
                };

                let class = AmdgpuInlineAsmRegClass::parse_with_prefix(prefix, 32)?;
                if start >= class.max_num() {
                    return Err("too large register for this class");
                }

                Ok(start)
            };

            let name = &name[1..];
            let class;
            let range = if let Some(name) = name.strip_suffix(".l") {
                class = AmdgpuInlineAsmRegClass::parse_with_prefix(prefix, 16)?;
                if matches!(class, AmdgpuInlineAsmRegClass::Sgpr(_)) {
                    return Err("invalid 16-bit SGPR register");
                }
                AmdgpuRegStart::Low(parse_num(name)?)
            } else if let Some(name) = name.strip_suffix(".h") {
                class = AmdgpuInlineAsmRegClass::parse_with_prefix(prefix, 16)?;
                if matches!(class, AmdgpuInlineAsmRegClass::Sgpr(_)) {
                    return Err("invalid 16-bit SGPR register");
                }
                AmdgpuRegStart::High(parse_num(name)?)
            } else {
                class = AmdgpuInlineAsmRegClass::parse_with_prefix(prefix, 32)?;
                let start = parse_num(name)?;
                AmdgpuRegStart::Full(start)
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
            if class == AmdgpuInlineAsmRegClass::Vgpr(16) {
                for i in 0..(class.max_num() / 2) {
                    set.insert(InlineAsmReg::Amdgpu(AmdgpuInlineAsmReg {
                        class,
                        range: AmdgpuRegStart::Low(i),
                    }));
                    set.insert(InlineAsmReg::Amdgpu(AmdgpuInlineAsmReg {
                        class,
                        range: AmdgpuRegStart::High(i),
                    }));
                }
            } else {
                for i in 0..class.max_num() {
                    set.insert(InlineAsmReg::Amdgpu(AmdgpuInlineAsmReg {
                        class,
                        range: AmdgpuRegStart::Full(i),
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
        if matches!(self.class, AmdgpuInlineAsmRegClass::Vgpr(_)) {
            // Overlapping 16-bit registers (not supported for sgprs)
            if let AmdgpuRegStart::Full(start) = self.range {
                for i in start..(start + self.class.bytes().div_ceil(4) - 1) {
                    cb(AmdgpuInlineAsmReg {
                        class: AmdgpuInlineAsmRegClass::Vgpr(16),
                        range: AmdgpuRegStart::Low(i),
                    });
                    cb(AmdgpuInlineAsmReg {
                        class: AmdgpuInlineAsmRegClass::Vgpr(16),
                        range: AmdgpuRegStart::High(i),
                    });
                }
            }
        }

        // Overlapping 32-bit registers, up to size 32
        for size in 1..=32 {
            let (AmdgpuRegStart::Low(start)
            | AmdgpuRegStart::High(start)
            | AmdgpuRegStart::Full(start)) = self.range;

            let size_range = size - 1;
            for overlap_start in (start - size_range)..=(start + self.class.bytes().div_ceil(4) - 1)
            {
                let class = match self.class {
                    AmdgpuInlineAsmRegClass::Sgpr(_) => AmdgpuInlineAsmRegClass::Sgpr(size * 32),
                    AmdgpuInlineAsmRegClass::Vgpr(_) => AmdgpuInlineAsmRegClass::Vgpr(size * 32),
                };
                cb(AmdgpuInlineAsmReg { class, range: AmdgpuRegStart::Full(overlap_start) });
            }
        }
    }
}
