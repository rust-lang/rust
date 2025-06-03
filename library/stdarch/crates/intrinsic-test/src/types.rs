use std::fmt;
use std::str::FromStr;

use itertools::Itertools as _;

use crate::Language;
use crate::format::Indentation;
use crate::values::value_for_array;

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
            _ => Err(format!("Impossible to parse argument kind {s}")),
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
        match self {
            IntrinsicType::Ptr { child, .. } => child.inner_size(),
            IntrinsicType::Type {
                bit_len: Some(bl), ..
            } => *bl,
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

    pub fn num_vectors(&self) -> u32 {
        match *self {
            IntrinsicType::Ptr { ref child, .. } => child.num_vectors(),
            IntrinsicType::Type {
                vec_len: Some(vl), ..
            } => vl,
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

    pub fn c_scalar_type(&self) -> String {
        format!(
            "{prefix}{bits}_t",
            prefix = self.kind().c_prefix(),
            bits = self.inner_size()
        )
    }

    pub fn rust_scalar_type(&self) -> String {
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
            } => format!("{}{bit_len}x{simd_len}_t", kind.c_prefix()),
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                simd_len: Some(simd_len),
                vec_len: Some(vec_len),
                ..
            } => format!("{}{bit_len}x{simd_len}x{vec_len}_t", kind.c_prefix()),
            _ => todo!("{:#?}", self),
        }
    }

    pub fn c_single_vector_type(&self) -> String {
        match self {
            IntrinsicType::Ptr { child, .. } => child.c_single_vector_type(),
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                simd_len: Some(simd_len),
                vec_len: Some(_),
                ..
            } => format!("{}{bit_len}x{simd_len}_t", kind.c_prefix()),
            _ => unreachable!("Shouldn't be called on this type"),
        }
    }

    pub fn rust_type(&self) -> String {
        match self {
            IntrinsicType::Ptr { child, .. } => child.c_type(),
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                simd_len: None,
                vec_len: None,
                ..
            } => format!("{}{bit_len}", kind.rust_prefix()),
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                simd_len: Some(simd_len),
                vec_len: None,
                ..
            } => format!("{}{bit_len}x{simd_len}_t", kind.c_prefix()),
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                simd_len: Some(simd_len),
                vec_len: Some(vec_len),
                ..
            } => format!("{}{bit_len}x{simd_len}x{vec_len}_t", kind.c_prefix()),
            _ => todo!("{:#?}", self),
        }
    }

    /// Gets a cast for this type if needs promotion.
    /// This is required for 8 bit types due to printing as the 8 bit types use
    /// a char and when using that in `std::cout` it will print as a character,
    /// which means value of 0 will be printed as a null byte.
    ///
    /// This is also needed for polynomial types because we want them to be
    /// printed as unsigned integers to match Rust's `Debug` impl.
    pub fn c_promotion(&self) -> &str {
        match *self {
            IntrinsicType::Type {
                kind,
                bit_len: Some(8),
                ..
            } => match kind {
                TypeKind::Int => "(int)",
                TypeKind::UInt => "(unsigned int)",
                TypeKind::Poly => "(unsigned int)(uint8_t)",
                _ => "",
            },
            IntrinsicType::Type {
                kind: TypeKind::Poly,
                bit_len: Some(bit_len),
                ..
            } => match bit_len {
                8 => unreachable!("handled above"),
                16 => "(uint16_t)",
                32 => "(uint32_t)",
                64 => "(uint64_t)",
                128 => "",
                _ => panic!("invalid bit_len"),
            },
            _ => "",
        }
    }

    /// Generates an initialiser for an array, which can be used to initialise an argument for the
    /// intrinsic call.
    ///
    /// This is determistic based on the pass number.
    ///
    /// * `loads`: The number of values that need to be loaded from the argument array
    /// * e.g for argument type uint32x2, loads=2 results in a string representing 4 32-bit values
    ///
    /// Returns a string such as
    /// * `{0x1, 0x7F, 0xFF}` if `language` is `Language::C`
    /// * `[0x1 as _, 0x7F as _, 0xFF as _]` if `language` is `Language::Rust`
    pub fn populate_random(
        &self,
        indentation: Indentation,
        loads: u32,
        language: &Language,
    ) -> String {
        match self {
            IntrinsicType::Ptr { child, .. } => child.populate_random(indentation, loads, language),
            IntrinsicType::Type {
                bit_len: Some(bit_len @ (8 | 16 | 32 | 64)),
                kind: kind @ (TypeKind::Int | TypeKind::UInt | TypeKind::Poly),
                simd_len,
                vec_len,
                ..
            } => {
                let (prefix, suffix) = match language {
                    Language::Rust => ("[", "]"),
                    Language::C => ("{", "}"),
                };
                let body_indentation = indentation.nested();
                format!(
                    "{prefix}\n{body}\n{indentation}{suffix}",
                    body = (0..(simd_len.unwrap_or(1) * vec_len.unwrap_or(1) + loads - 1))
                        .format_with(",\n", |i, fmt| {
                            let src = value_for_array(*bit_len, i);
                            assert!(src == 0 || src.ilog2() < *bit_len);
                            if *kind == TypeKind::Int && (src >> (*bit_len - 1)) != 0 {
                                // `src` is a two's complement representation of a negative value.
                                let mask = !0u64 >> (64 - *bit_len);
                                let ones_compl = src ^ mask;
                                let twos_compl = ones_compl + 1;
                                if (twos_compl == src) && (language == &Language::C) {
                                    // `src` is INT*_MIN. C requires `-0x7fffffff - 1` to avoid
                                    // undefined literal overflow behaviour.
                                    fmt(&format_args!("{body_indentation}-{ones_compl:#x} - 1"))
                                } else {
                                    fmt(&format_args!("{body_indentation}-{twos_compl:#x}"))
                                }
                            } else {
                                fmt(&format_args!("{body_indentation}{src:#x}"))
                            }
                        })
                )
            }
            IntrinsicType::Type {
                kind: TypeKind::Float,
                bit_len: Some(bit_len @ (16 | 32 | 64)),
                simd_len,
                vec_len,
                ..
            } => {
                let (prefix, cast_prefix, cast_suffix, suffix) = match (language, bit_len) {
                    (&Language::Rust, 16) => ("[", "f16::from_bits(", ")", "]"),
                    (&Language::Rust, 32) => ("[", "f32::from_bits(", ")", "]"),
                    (&Language::Rust, 64) => ("[", "f64::from_bits(", ")", "]"),
                    (&Language::C, 16) => ("{", "cast<float16_t, uint16_t>(", ")", "}"),
                    (&Language::C, 32) => ("{", "cast<float, uint32_t>(", ")", "}"),
                    (&Language::C, 64) => ("{", "cast<double, uint64_t>(", ")", "}"),
                    _ => unreachable!(),
                };
                format!(
                    "{prefix}\n{body}\n{indentation}{suffix}",
                    body = (0..(simd_len.unwrap_or(1) * vec_len.unwrap_or(1) + loads - 1))
                        .format_with(",\n", |i, fmt| fmt(&format_args!(
                            "{indentation}{cast_prefix}{src:#x}{cast_suffix}",
                            indentation = indentation.nested(),
                            src = value_for_array(*bit_len, i)
                        )))
                )
            }
            _ => unimplemented!("populate random: {:#?}", self),
        }
    }

    /// Determines the load function for this type.
    pub fn get_load_function(&self, armv7_p64_workaround: bool) -> String {
        match self {
            IntrinsicType::Ptr { child, .. } => child.get_load_function(armv7_p64_workaround),
            IntrinsicType::Type {
                kind: k,
                bit_len: Some(bl),
                simd_len,
                vec_len,
                ..
            } => {
                let quad = if simd_len.unwrap_or(1) * bl > 64 {
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
                        // The ACLE doesn't support 64-bit polynomial loads on Armv7
                        TypeKind::Poly => if armv7_p64_workaround && *bl == 64 {"s"} else {"p"},
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

    pub fn from_c(s: &str) -> Result<IntrinsicType, String> {
        const CONST_STR: &str = "const";
        if let Some(s) = s.strip_suffix('*') {
            let (s, constant) = match s.trim().strip_suffix(CONST_STR) {
                Some(stripped) => (stripped, true),
                None => (s, false),
            };
            let s = s.trim_end();
            Ok(IntrinsicType::Ptr {
                constant,
                child: Box::new(IntrinsicType::from_c(s)?),
            })
        } else {
            // [const ]TYPE[{bitlen}[x{simdlen}[x{vec_len}]]][_t]
            let (mut s, constant) = match s.strip_prefix(CONST_STR) {
                Some(stripped) => (stripped.trim(), true),
                None => (s, false),
            };
            s = s.strip_suffix("_t").unwrap_or(s);
            let mut parts = s.split('x'); // [[{bitlen}], [{simdlen}], [{vec_len}] ]
            let start = parts.next().ok_or("Impossible to parse type")?;
            if let Some(digit_start) = start.find(|c: char| c.is_ascii_digit()) {
                let (arg_kind, bit_len) = start.split_at(digit_start);
                let arg_kind = arg_kind.parse::<TypeKind>()?;
                let bit_len = bit_len.parse::<u32>().map_err(|err| err.to_string())?;
                let simd_len = match parts.next() {
                    Some(part) => Some(
                        part.parse::<u32>()
                            .map_err(|_| "Couldn't parse simd_len: {part}")?,
                    ),
                    None => None,
                };
                let vec_len = match parts.next() {
                    Some(part) => Some(
                        part.parse::<u32>()
                            .map_err(|_| "Couldn't parse vec_len: {part}")?,
                    ),
                    None => None,
                };
                Ok(IntrinsicType::Type {
                    constant,
                    kind: arg_kind,
                    bit_len: Some(bit_len),
                    simd_len,
                    vec_len,
                })
            } else {
                let kind = start.parse::<TypeKind>()?;
                let bit_len = match kind {
                    TypeKind::Int => Some(32),
                    _ => None,
                };
                Ok(IntrinsicType::Type {
                    constant,
                    kind: start.parse::<TypeKind>()?,
                    bit_len,
                    simd_len: None,
                    vec_len: None,
                })
            }
        }
    }
}
