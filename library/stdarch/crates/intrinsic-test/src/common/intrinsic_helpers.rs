use std::fmt;
use std::ops::Deref;
use std::str::FromStr;

use itertools::Itertools as _;

use super::cli::Language;
use super::indentation::Indentation;
use super::values::value_for_array;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Sign {
    Signed,
    Unsigned,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum TypeKind {
    BFloat,
    Float,
    Int(Sign),
    Char(Sign),
    Poly,
    Void,
    Mask,
    Vector,
}

impl FromStr for TypeKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bfloat" | "BF16" => Ok(Self::BFloat),
            "float" | "double" | "FP16" | "FP32" | "FP64" => Ok(Self::Float),
            "int" | "long" | "short" | "SI8" | "SI16" | "SI32" | "SI64" => {
                Ok(Self::Int(Sign::Signed))
            }
            "poly" => Ok(Self::Poly),
            "char" => Ok(Self::Char(Sign::Signed)),
            "uint" | "unsigned" | "UI8" | "UI16" | "UI32" | "UI64" => Ok(Self::Int(Sign::Unsigned)),
            "void" => Ok(Self::Void),
            "MASK" => Ok(Self::Mask),
            "M64" | "M128" | "M256" | "M512" => Ok(Self::Vector),
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
                Self::Int(Sign::Signed) => "int",
                Self::Int(Sign::Unsigned) => "uint",
                Self::Poly => "poly",
                Self::Void => "void",
                Self::Char(Sign::Signed) => "char",
                Self::Char(Sign::Unsigned) => "unsigned char",
                Self::Mask => "mask",
                Self::Vector => "vector",
            }
        )
    }
}

impl TypeKind {
    /// Gets the type part of a c typedef for a type that's in the form of {type}{size}_t.
    pub fn c_prefix(&self) -> &str {
        match self {
            Self::Float => "float",
            Self::Int(Sign::Signed) => "int",
            Self::Int(Sign::Unsigned) => "uint",
            Self::Poly => "poly",
            Self::Char(Sign::Signed) => "char",
            _ => unreachable!("Not used: {:#?}", self),
        }
    }

    /// Gets the rust prefix for the type kind i.e. i, u, f.
    pub fn rust_prefix(&self) -> &str {
        match self {
            Self::BFloat => "bf",
            Self::Float => "f",
            Self::Int(Sign::Signed) => "i",
            Self::Int(Sign::Unsigned) => "u",
            Self::Poly => "u",
            Self::Char(Sign::Unsigned) => "u",
            Self::Char(Sign::Signed) => "i",
            _ => unreachable!("Unused type kind: {:#?}", self),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct IntrinsicType {
    pub constant: bool,

    /// whether this object is a const pointer
    pub ptr_constant: bool,

    pub ptr: bool,

    pub kind: TypeKind,
    /// The bit length of this type (e.g. 32 for u32).
    pub bit_len: Option<u32>,

    /// Length of the SIMD vector (i.e. 4 for uint32x4_t), A value of `None`
    /// means this is not a simd type. A `None` can be assumed to be 1,
    /// although in some places a distinction is needed between `u64` and
    /// `uint64x1_t` this signals that.
    pub simd_len: Option<u32>,

    /// The number of rows for SIMD matrices (i.e. 2 for uint8x8x2_t).
    /// A value of `None` represents a type that does not contain any
    /// rows encoded in the type (e.g. uint8x8_t).
    /// A value of `None` can be assumed to be 1 though.
    pub vec_len: Option<u32>,
}

impl IntrinsicType {
    pub fn kind(&self) -> TypeKind {
        self.kind
    }

    pub fn inner_size(&self) -> u32 {
        if let Some(bl) = self.bit_len {
            bl
        } else {
            unreachable!("")
        }
    }

    pub fn num_lanes(&self) -> u32 {
        self.simd_len.unwrap_or(1)
    }

    pub fn num_vectors(&self) -> u32 {
        self.vec_len.unwrap_or(1)
    }

    pub fn is_simd(&self) -> bool {
        self.simd_len.is_some() || self.vec_len.is_some()
    }

    pub fn is_ptr(&self) -> bool {
        self.ptr
    }

    pub fn c_scalar_type(&self) -> String {
        match self.kind() {
            TypeKind::Char(_) => String::from("char"),
            _ => format!(
                "{prefix}{bits}_t",
                prefix = self.kind().c_prefix(),
                bits = self.inner_size()
            ),
        }
    }

    pub fn rust_scalar_type(&self) -> String {
        format!(
            "{prefix}{bits}",
            prefix = self.kind().rust_prefix(),
            bits = self.inner_size()
        )
    }

    pub fn c_promotion(&self) -> &str {
        match *self {
            IntrinsicType {
                kind,
                bit_len: Some(8),
                ..
            } => match kind {
                TypeKind::Int(Sign::Signed) => "(int)",
                TypeKind::Int(Sign::Unsigned) => "(unsigned int)",
                TypeKind::Poly => "(unsigned int)(uint8_t)",
                _ => "",
            },
            IntrinsicType {
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
            IntrinsicType {
                kind: TypeKind::Float,
                bit_len: Some(bit_len),
                ..
            } => match bit_len {
                16 => "(float16_t)",
                32 => "(float)",
                64 => "(double)",
                128 => "",
                _ => panic!("invalid bit_len"),
            },
            IntrinsicType {
                kind: TypeKind::Char(_),
                ..
            } => "(char)",
            _ => "",
        }
    }

    pub fn populate_random(
        &self,
        indentation: Indentation,
        loads: u32,
        language: &Language,
    ) -> String {
        match self {
            IntrinsicType {
                bit_len: Some(bit_len @ (8 | 16 | 32 | 64)),
                kind: kind @ (TypeKind::Int(_) | TypeKind::Poly | TypeKind::Char(_)),
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
                            if *kind == TypeKind::Int(Sign::Signed) && (src >> (*bit_len - 1)) != 0
                            {
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
            IntrinsicType {
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

    pub fn is_rust_vals_array_const(&self) -> bool {
        match self {
            // Floats have to be loaded at runtime for stable NaN conversion.
            IntrinsicType {
                kind: TypeKind::Float,
                ..
            } => false,
            IntrinsicType {
                kind: TypeKind::Int(_) | TypeKind::Poly,
                ..
            } => true,
            _ => unimplemented!(),
        }
    }

    pub fn as_call_param_c(&self, name: &String) -> String {
        if self.ptr {
            format!("&{name}")
        } else {
            name.clone()
        }
    }
}

pub trait IntrinsicTypeDefinition: Deref<Target = IntrinsicType> {
    /// Determines the load function for this type.
    /// can be implemented in an `impl` block
    fn get_load_function(&self, _language: Language) -> String;

    /// can be implemented in an `impl` block
    fn get_lane_function(&self) -> String;

    /// Gets a string containing the typename for this type in C format.
    /// can be directly defined in `impl` blocks
    fn c_type(&self) -> String;

    /// can be directly defined in `impl` blocks
    fn c_single_vector_type(&self) -> String;

    /// Generates a std::cout for the intrinsics results that will match the
    /// rust debug output format for the return type. The generated line assumes
    /// there is an int i in scope which is the current pass number.
    fn print_result_c(&self, indentation: Indentation, additional: &str) -> String;
}
