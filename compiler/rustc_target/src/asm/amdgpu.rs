use std::fmt;

use rustc_span::Symbol;

use super::{InlineAsmArch, InlineAsmType, ModifierInfo};

def_reg_class! {
    Amdgpu AmdgpuInlineAsmRegClass {
        sgpr,
        vgpr,
    }
}

// See https://llvm.org/docs/AMDGPUOperandSyntax.html
impl AmdgpuInlineAsmRegClass {
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
        types! { _: I16, F16, I32, F32, I64, F64, I128; }
    }

    /// The number of supported registers in this class.
    /// The returned number is the length, so supported register
    /// indices are 0 to max_num()-1.
    fn max_num(self) -> u32 {
        match self {
            Self::sgpr => 106,
            Self::vgpr => 256,
        }
    }

    /// Prefix when printed and register constraint in LLVM.
    pub fn prefix(self) -> &'static str {
        match self {
            Self::sgpr => "s",
            Self::vgpr => "v",
        }
    }

    /// Get register class from prefix.
    fn parse_prefix(prefix: char) -> Result<Self, &'static str> {
        match prefix {
            's' => Ok(Self::sgpr),
            'v' => Ok(Self::vgpr),
            _ => Err("unknown register prefix"),
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
    class: AmdgpuInlineAsmRegClass,
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

    pub fn reg_class(self) -> AmdgpuInlineAsmRegClass {
        self.class
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

                if end >= class.max_num() {
                    return Err("too large register for this class");
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

                if start >= class.max_num() {
                    return Err("too large register for this class");
                }

                Ok(start)
            };

            let name = &name[1..];
            let range = if let Some(name) = name.strip_suffix(".l") {
                AmdgpuRegRange::Low(parse_num(name)?)
            } else if let Some(name) = name.strip_suffix(".h") {
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

    // Add single registers of each class (no register ranges)
    #[allow(rustc::potential_query_instability)]
    for class in regclass_map().keys() {
        let InlineAsmRegClass::Amdgpu(class) = *class else { unreachable!("Must be amdgpu class") };
        if let Some(set) = map.get_mut(&InlineAsmRegClass::Amdgpu(class)) {
            for i in 0..class.max_num() {
                set.insert(InlineAsmReg::Amdgpu(AmdgpuInlineAsmReg {
                    class,
                    range: AmdgpuRegRange::Range { start: i, end: i },
                }));
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

    // There are too many conflicts to list
    pub fn overlapping_regs(self, mut _cb: impl FnMut(AmdgpuInlineAsmReg)) {}
}
