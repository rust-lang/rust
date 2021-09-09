use regex::Regex;
use serde::{Deserialize, Deserializer};
use std::fmt;
use std::str::FromStr;

use crate::values::values_for_pass;
use crate::Language;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum TypeKind {
    BFloat,
    Float,
    Int,
    UInt,
    Poly,
    Void,
}

impl FromStr for TypeKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bfloat" => Ok(Self::BFloat),
            "float" => Ok(Self::Float),
            "int" => Ok(Self::Int),
            "poly" => Ok(Self::Poly),
            "uint" | "unsigned" => Ok(Self::UInt),
            "void" => Ok(Self::Void),
            _ => Err(format!("Impossible to parse argument kind {}", s)),
        }
    }
}

impl fmt::Display for TypeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::BFloat => "bfloat",
                Self::Float => "float",
                Self::Int => "int",
                Self::UInt => "uint",
                Self::Poly => "poly",
                Self::Void => "void",
            }
        )
    }
}

impl TypeKind {
    /// Gets the type part of a c typedef for a type that's in the form of {type}{size}_t.
    pub fn c_prefix(&self) -> &str {
        match self {
            Self::Float => "float",
            Self::Int => "int",
            Self::UInt => "uint",
            Self::Poly => "poly",
            _ => unreachable!("Not used: {:#?}", self),
        }
    }

    /// Gets the rust prefix for the type kind i.e. i, u, f.
    pub fn rust_prefix(&self) -> &str {
        match self {
            Self::Float => "f",
            Self::Int => "i",
            Self::UInt => "u",
            Self::Poly => "u",
            _ => unreachable!("Unused type kind: {:#?}", self),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum IntrinsicType {
    Ptr {
        constant: bool,
        child: Box<IntrinsicType>,
    },
    Type {
        constant: bool,
        kind: TypeKind,
        /// The bit length of this type (e.g. 32 for u32).
        bit_len: Option<u32>,

        /// Length of the SIMD vector (i.e. 4 for uint32x4_t), A value of `None`
        /// means this is not a simd type. A `None` can be assumed to be 1,
        /// although in some places a distinction is needed between `u64` and
        /// `uint64x1_t` this signals that.
        simd_len: Option<u32>,

        /// The number of rows for SIMD matrices (i.e. 2 for uint8x8x2_t).
        /// A value of `None` represents a type that does not contain any
        /// rows encoded in the type (e.g. uint8x8_t).
        /// A value of `None` can be assumed to be 1 though.
        vec_len: Option<u32>,
    },
}

impl IntrinsicType {
    /// Get the TypeKind for this type, recursing into pointers.
    pub fn kind(&self) -> TypeKind {
        match *self {
            IntrinsicType::Ptr { ref child, .. } => child.kind(),
            IntrinsicType::Type { kind, .. } => kind,
        }
    }

    /// Get the size of a single element inside this type, recursing into
    /// pointers, i.e. a pointer to a u16 would be 16 rather than the size
    /// of a pointer.
    pub fn inner_size(&self) -> u32 {
        match *self {
            IntrinsicType::Ptr { ref child, .. } => child.inner_size(),
            IntrinsicType::Type {
                bit_len: Some(bl), ..
            } => bl,
            _ => unreachable!(""),
        }
    }

    pub fn num_lanes(&self) -> u32 {
        match *self {
            IntrinsicType::Ptr { ref child, .. } => child.num_lanes(),
            IntrinsicType::Type {
                simd_len: Some(sl), ..
            } => sl,
            _ => 1,
        }
    }

    /// Determine if the type is a simd type, this will treat a type such as
    /// `uint64x1` as simd.
    pub fn is_simd(&self) -> bool {
        match *self {
            IntrinsicType::Ptr { ref child, .. } => child.is_simd(),
            IntrinsicType::Type {
                simd_len: None,
                vec_len: None,
                ..
            } => false,
            _ => true,
        }
    }

    pub fn is_ptr(&self) -> bool {
        match *self {
            IntrinsicType::Ptr { .. } => true,
            IntrinsicType::Type { .. } => false,
        }
    }

    pub fn from_rust(ty: &str) -> Result<Self, String> {
        lazy_static! {
            static ref SIMD_TYPE: Regex = Regex::new(r#"([a-z]*)([0-9]*)x([0-9]*)_t"#).unwrap();
            static ref MULTI_SIMD_TYPE: Regex =
                Regex::new(r#"([a-z]*)([0-9]*)x([0-9]*)x([0-9]*)_t"#).unwrap();
            static ref RUST_TYPE: Regex = Regex::new(r#"([iuf]|float|poly)([0-9]+)"#).unwrap();
        }

        debug!("Parsing type: {}", ty);

        if let Some(ty) = ty.strip_prefix('*') {
            let (constant, ty) = if let Some(ty) = ty.strip_prefix("const") {
                (true, ty.trim())
            } else if let Some(ty) = ty.strip_prefix("mut") {
                (false, ty.trim())
            } else {
                (false, ty)
            };
            return Ok(Self::Ptr {
                constant,
                child: Box::new(Self::from_rust(ty)?),
            });
        }

        let (constant, ty) = if let Some(ty) = ty.strip_prefix("const") {
            (true, ty.trim())
        } else {
            (false, ty)
        };

        if let Some(captures) = MULTI_SIMD_TYPE.captures(ty) {
            let kind = captures
                .get(1)
                .map(|s| s.as_str().parse::<TypeKind>().unwrap())
                .unwrap();
            let bit_len = captures.get(2).map(|s| s.as_str().parse::<u32>().unwrap());
            let simd_len = captures.get(3).map(|s| s.as_str().parse::<u32>().unwrap());
            let vec_len = captures.get(4).map(|s| s.as_str().parse::<u32>().unwrap());
            Ok(Self::Type {
                constant,
                kind,
                bit_len,
                simd_len,
                vec_len,
            })
        } else if let Some(captures) = SIMD_TYPE.captures(ty) {
            let kind = captures
                .get(1)
                .map(|s| s.as_str().parse::<TypeKind>().unwrap())
                .unwrap();
            let bit_len = captures.get(2).map(|s| s.as_str().parse::<u32>().unwrap());
            let simd_len = captures.get(3).map(|s| s.as_str().parse::<u32>().unwrap());

            Ok(Self::Type {
                constant,
                kind,
                bit_len,
                simd_len,
                vec_len: None,
            })
        } else if let Some(captures) = RUST_TYPE.captures(ty) {
            let kind = captures
                .get(1)
                .map(|s| match s.as_str() {
                    "i" => TypeKind::Int,
                    "u" => TypeKind::UInt,
                    "f" => TypeKind::Float,
                    "float" => TypeKind::Float,
                    "poly" => TypeKind::Poly,
                    a => panic!("Unexpected type: {} found", a),
                })
                .unwrap();
            let bit_len = captures.get(2).map(|s| s.as_str().parse::<u32>().unwrap());
            Ok(Self::Type {
                constant,
                kind,
                bit_len,
                simd_len: None,
                vec_len: None,
            })
        } else {
            match ty {
                "int" => Ok(Self::Type {
                    constant,
                    kind: TypeKind::Int,
                    bit_len: Some(32),
                    simd_len: None,
                    vec_len: None,
                }),
                "void" => Ok(Self::Type {
                    constant: false,
                    kind: TypeKind::Void,
                    bit_len: None,
                    simd_len: None,
                    vec_len: None,
                }),
                _ => Err(format!("Failed to parse type: {}", ty)),
            }
        }
    }

    #[allow(unused)]
    fn c_scalar_type(&self) -> String {
        format!(
            "{prefix}{bits}_t",
            prefix = self.kind().c_prefix(),
            bits = self.inner_size()
        )
    }

    fn rust_scalar_type(&self) -> String {
        format!(
            "{prefix}{bits}",
            prefix = self.kind().rust_prefix(),
            bits = self.inner_size()
        )
    }

    /// Gets a string containing the typename for this type in C format.
    pub fn c_type(&self) -> String {
        match self {
            IntrinsicType::Ptr { child, .. } => child.c_type(),
            IntrinsicType::Type {
                constant,
                kind,
                bit_len: Some(bit_len),
                simd_len: None,
                vec_len: None,
                ..
            } => format!(
                "{}{}{}_t",
                if *constant { "const " } else { "" },
                kind.c_prefix(),
                bit_len
            ),
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                simd_len: Some(simd_len),
                vec_len: None,
                ..
            } => format!("{}{}x{}_t", kind.c_prefix(), bit_len, simd_len),
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                simd_len: Some(simd_len),
                vec_len: Some(vec_len),
                ..
            } => format!("{}{}x{}x{}_t", kind.c_prefix(), bit_len, simd_len, vec_len),
            _ => todo!("{:#?}", self),
        }
    }

    /// Gets a cast for this type if needs promotion.
    /// This is required for 8 bit types due to printing as the 8 bit types use
    /// a char and when using that in `std::cout` it will print as a character,
    /// which means value of 0 will be printed as a null byte.
    pub fn c_promotion(&self) -> &str {
        match *self {
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                ..
            } if bit_len == 8 => match kind {
                TypeKind::Int => "(int)",
                TypeKind::UInt => "(unsigned int)",
                TypeKind::Poly => "(unsigned int)",
                _ => "",
            },
            _ => "",
        }
    }

    /// Generates a comma list of values that can be used to initialize an
    /// argument for the intrinsic call.
    /// This is determistic based on the pass number.
    ///
    /// * `pass`: The pass index, i.e. the iteration index for the call to an intrinsic
    ///
    /// Returns a string such as
    /// * `0x1, 0x7F, 0xFF` if `language` is `Language::C`
    /// * `0x1 as _, 0x7F as _, 0xFF as _` if `language` is `Language::Rust`
    pub fn populate_random(&self, pass: usize, language: &Language) -> String {
        match self {
            IntrinsicType::Ptr { child, .. } => child.populate_random(pass, language),
            IntrinsicType::Type {
                bit_len: Some(bit_len),
                kind,
                simd_len,
                vec_len,
                ..
            } if kind == &TypeKind::Int || kind == &TypeKind::UInt || kind == &TypeKind::Poly => (0
                ..(simd_len.unwrap_or(1) * vec_len.unwrap_or(1)))
                .map(|i| {
                    format!(
                        "{}{}",
                        values_for_pass(*bit_len, i, pass),
                        match language {
                            &Language::Rust => format!(" as {ty} ", ty = self.rust_scalar_type()),
                            &Language::C => String::from(""),
                        }
                    )
                })
                .collect::<Vec<_>>()
                .join(","),
            IntrinsicType::Type {
                kind: TypeKind::Float,
                bit_len: Some(32),
                simd_len,
                vec_len,
                ..
            } => (0..(simd_len.unwrap_or(1) * vec_len.unwrap_or(1)))
                .map(|i| {
                    format!(
                        "{}({})",
                        match language {
                            &Language::Rust => "f32::from_bits",
                            &Language::C => "cast<float, uint32_t>",
                        },
                        values_for_pass(32, i, pass),
                    )
                })
                .collect::<Vec<_>>()
                .join(","),
            IntrinsicType::Type {
                kind: TypeKind::Float,
                bit_len: Some(64),
                simd_len,
                vec_len,
                ..
            } => (0..(simd_len.unwrap_or(1) * vec_len.unwrap_or(1)))
                .map(|i| {
                    format!(
                        "{}({}{})",
                        match language {
                            &Language::Rust => "f64::from_bits",
                            &Language::C => "cast<double, uint64_t>",
                        },
                        values_for_pass(64, i, pass),
                        match language {
                            &Language::Rust => " as u64",
                            &Language::C => "",
                        }
                    )
                })
                .collect::<Vec<_>>()
                .join(","),
            _ => unreachable!("populate random: {:#?}", self),
        }
    }

    /// Determines the load function for this type.
    #[allow(unused)]
    pub fn get_load_function(&self) -> String {
        match self {
            IntrinsicType::Ptr { child, .. } => child.get_load_function(),
            IntrinsicType::Type {
                kind: k,
                bit_len: Some(bl),
                simd_len,
                vec_len,
                ..
            } => {
                let quad = if (simd_len.unwrap_or(1) * bl) > 64 {
                    "q"
                } else {
                    ""
                };
                format!(
                    "vld{len}{quad}_{type}{size}",
                    type = match k {
                        TypeKind::UInt => "u",
                        TypeKind::Int => "s",
                        TypeKind::Float => "f",
                        TypeKind::Poly => "p",
                        x => todo!("get_load_function TypeKind: {:#?}", x),
                    },
                    size = bl,
                    quad = quad,
                    len = vec_len.unwrap_or(1),
                )
            }
            _ => todo!("get_load_function IntrinsicType: {:#?}", self),
        }
    }

    /// Determines the get lane function for this type.
    pub fn get_lane_function(&self) -> String {
        match self {
            IntrinsicType::Ptr { child, .. } => child.get_lane_function(),
            IntrinsicType::Type {
                kind: k,
                bit_len: Some(bl),
                simd_len,
                ..
            } => {
                let quad = if (simd_len.unwrap_or(1) * bl) > 64 {
                    "q"
                } else {
                    ""
                };
                format!(
                    "vget{quad}_lane_{type}{size}",
                    type = match k {
                        TypeKind::UInt => "u",
                        TypeKind::Int => "s",
                        TypeKind::Float => "f",
                        TypeKind::Poly => "p",
                        x => todo!("get_load_function TypeKind: {:#?}", x),
                    },
                    size = bl,
                    quad = quad,
                )
            }
            _ => todo!("get_lane_function IntrinsicType: {:#?}", self),
        }
    }
}

impl<'de> Deserialize<'de> for IntrinsicType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        let s = String::deserialize(deserializer)?;
        Self::from_rust(&s).map_err(Error::custom)
    }
}
