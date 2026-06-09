use std::cmp;
use std::fmt;
use std::ops::Deref;
use std::str::FromStr;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Sign {
    Signed,
    Unsigned,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum TypeKind {
    Bool,
    BFloat,
    Float,
    Int(Sign),
    Char(Sign),
    Poly,
    Void,
    Mask,
    Vector,
    SvPattern,
    SvPrefetchOp,
}

impl FromStr for TypeKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "svbool" | "bool" => Ok(Self::Bool),
            "svbfloat" | "bfloat" | "BF16" => Ok(Self::BFloat),
            "svfloat" | "float" | "double" | "FP16" | "FP32" | "FP64" => Ok(Self::Float),
            "svint" | "int" | "long" | "short" | "SI8" | "SI16" | "SI32" | "SI64" => {
                Ok(Self::Int(Sign::Signed))
            }
            "poly" => Ok(Self::Poly),
            "char" => Ok(Self::Char(Sign::Signed)),
            "svuint" | "uint" | "unsigned" | "UI8" | "UI16" | "UI32" | "UI64" => {
                Ok(Self::Int(Sign::Unsigned))
            }
            "void" => Ok(Self::Void),
            "MASK" => Ok(Self::Mask),
            "M128" | "M256" | "M512" => Ok(Self::Vector),
            "svpattern" => Ok(Self::SvPattern),
            "svprfop" => Ok(Self::SvPrefetchOp),
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
                Self::Bool => "bool",
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
                Self::SvPattern => "svpattern",
                Self::SvPrefetchOp => "svprfop",
            }
        )
    }
}

impl TypeKind {
    /// Returns the type component of a C typedef for a type of the form of `{type}{size}_t`
    pub fn c_prefix(&self) -> &str {
        match self {
            Self::Bool => "bool",
            Self::Float => "float",
            Self::Int(Sign::Signed) => "int",
            Self::Int(Sign::Unsigned) => "uint",
            Self::Mask => "uint",
            Self::Poly => "poly",
            Self::Char(Sign::Signed) => "char",
            Self::Vector => "int",
            _ => unreachable!("Not used: {self:#?}"),
        }
    }

    /// Returns the Rust prefix for this type kind i.e. `i`, `u`, or `f`.
    pub fn rust_prefix(&self) -> &str {
        match self {
            Self::BFloat => "bf",
            Self::Float => "f",
            Self::Int(Sign::Signed) => "i",
            Self::Int(Sign::Unsigned) => "u",
            Self::Poly => "u",
            Self::Char(Sign::Unsigned) => "u",
            Self::Char(Sign::Signed) => "i",
            Self::Mask => "u",
            _ => unreachable!("Unused type kind: {self:#?}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimdLen {
    Scalable,
    Fixed(u32),
}

impl std::fmt::Display for SimdLen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalable => unimplemented!(),
            Self::Fixed(len) => <u32 as std::fmt::Display>::fmt(len, f),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct IntrinsicType {
    /// Is this an immediate?
    pub constant: bool,

    /// Is this is a const pointer to the type?
    pub ptr_constant: bool,

    /// Is this is a pointer to the type?
    pub ptr: bool,

    /// Element type (e.g. `TypeKind::Int(Sign::Unsigned)` for `uint64x2_t`).
    pub kind: TypeKind,

    /// Number of bits of this type (e.g. 32 for `u32`).
    pub bit_len: Option<u32>,

    /// Length of a SIMD vector (i.e. `Fixed(4)` for `uint32x4_t`).
    ///
    /// A value of `None` means this is not a SIMD type. The number of lanes of a type with
    /// `simd_len=None` can be assumed to be one, though it is important to maintain a distinction
    /// between `simd_len=None` and `simd_len=Some(Fixed(1))` so as to differentiate between `u64`
    /// and `uint64x1_t`. A value of `Some(Scalable)` indicates that this is a scalable vector.
    pub simd_len: Option<SimdLen>,

    /// Number of rows of a SIMD matrix (i.e. 2 for `uint8x8x2_t`).
    ///
    /// A value of `None` means this is not a SIMD matrix (e.g. `uint8x8_t`). The number of rows of
    /// a type with `vec_len=None` can be assumed to be one.
    pub vec_len: Option<u32>,
}

impl IntrinsicType {
    /// Returns the element type
    pub fn kind(&self) -> TypeKind {
        self.kind
    }

    /// Returns the number of bits of the type (with a minimum of `8`)
    pub fn inner_size(&self) -> u32 {
        if let Some(bl) = self.bit_len {
            cmp::max(bl, 8)
        } else {
            unreachable!("{self:#?}")
        }
    }

    /// Returns the number of lanes of the type
    pub fn num_lanes(&self) -> u32 {
        self.simd_len
            .as_ref()
            .map(|len| match len {
                SimdLen::Scalable => unimplemented!(),
                SimdLen::Fixed(len) => *len,
            })
            .unwrap_or(1)
    }

    /// Returns the number of vectors of the type
    pub fn num_vectors(&self) -> u32 {
        self.vec_len.unwrap_or(1)
    }

    /// Returns `true` if this represents a SIMD vector
    pub fn is_simd(&self) -> bool {
        self.simd_len.is_some() || self.vec_len.is_some()
    }

    /// Returns `true` if this is a pointer
    pub fn is_ptr(&self) -> bool {
        self.ptr
    }
}

pub trait IntrinsicTypeDefinition: Deref<Target = IntrinsicType> {
    /// Determines the load function for this type.
    fn get_load_function(&self) -> String;

    /// Gets a string containing the typename for this type in C.
    fn c_type(&self) -> String;

    /// Gets a string containing the typename for this type in Rust.
    fn rust_type(&self) -> String;

    /// Gets a string containing the name of the scalar type corresponding to this type if it is a
    /// vector.
    fn rust_scalar_type(&self) -> String {
        if self.is_simd() {
            format!(
                "{prefix}{bits}",
                prefix = self.kind().rust_prefix(),
                bits = self.inner_size()
            )
        } else {
            self.rust_type()
        }
    }
}
