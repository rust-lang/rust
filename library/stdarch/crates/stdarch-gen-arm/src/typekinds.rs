use proc_macro2::TokenStream;
use quote::{ToTokens, TokenStreamExt, quote};
use regex::Regex;
use serde_with::{DeserializeFromStr, SerializeDisplay};
use std::fmt;
use std::str::FromStr;
use std::sync::LazyLock;

use crate::context;
use crate::expression::{Expression, FnCall};
use crate::intrinsic::AccessLevel;
use crate::wildcards::Wildcard;

const VECTOR_FULL_REGISTER_SIZE: u32 = 128;
const VECTOR_HALF_REGISTER_SIZE: u32 = VECTOR_FULL_REGISTER_SIZE / 2;

#[derive(Debug, Clone, Copy)]
pub enum TypeRepr {
    C,
    Rust,
    LLVMMachine,
    ACLENotation,
    Size,
    SizeLiteral,
    TypeKind,
    SizeInBytesLog2,
}

pub trait ToRepr {
    fn repr(&self, repr: TypeRepr) -> String;

    fn c_repr(&self) -> String {
        self.repr(TypeRepr::C)
    }

    fn rust_repr(&self) -> String {
        self.repr(TypeRepr::Rust)
    }

    fn llvm_machine_repr(&self) -> String {
        self.repr(TypeRepr::LLVMMachine)
    }

    fn acle_notation_repr(&self) -> String {
        self.repr(TypeRepr::ACLENotation)
    }

    fn size(&self) -> String {
        self.repr(TypeRepr::Size)
    }

    fn size_literal(&self) -> String {
        self.repr(TypeRepr::SizeLiteral)
    }

    fn type_kind(&self) -> String {
        self.repr(TypeRepr::TypeKind)
    }

    fn size_in_bytes_log2(&self) -> String {
        self.repr(TypeRepr::SizeInBytesLog2)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct TypeKindOptions {
    f: bool,
    s: bool,
    u: bool,
    p: bool,
}

impl TypeKindOptions {
    pub fn contains(&self, kind: BaseTypeKind) -> bool {
        match kind {
            BaseTypeKind::Float => self.f,
            BaseTypeKind::Int => self.s,
            BaseTypeKind::UInt => self.u,
            BaseTypeKind::Poly => self.p,
            BaseTypeKind::Bool => false,
        }
    }
}

impl FromStr for TypeKindOptions {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut result = Self::default();
        for kind in s.bytes() {
            match kind {
                b'f' => result.f = true,
                b's' => result.s = true,
                b'u' => result.u = true,
                b'p' => result.p = true,
                _ => {
                    return Err(format!("unknown type kind: {}", char::from(kind)));
                }
            }
        }
        Ok(result)
    }
}

impl fmt::Display for TypeKindOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.f.then(|| write!(f, "f")).transpose()?;
        self.s.then(|| write!(f, "s")).transpose()?;
        self.u.then(|| write!(f, "u")).transpose().map(|_| ())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BaseTypeKind {
    Float,
    Int,
    UInt,
    Bool,
    Poly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BaseType {
    Sized(BaseTypeKind, u32),
    Unsized(BaseTypeKind),
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerializeDisplay, DeserializeFromStr,
)]
pub enum VectorTupleSize {
    Two,
    Three,
    Four,
}

impl VectorTupleSize {
    pub fn to_int(self) -> u32 {
        match self {
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VectorType {
    base_type: BaseType,
    lanes: u32,
    is_scalable: bool,
    tuple_size: Option<VectorTupleSize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, SerializeDisplay, DeserializeFromStr)]
pub enum TypeKind {
    Vector(VectorType),
    Base(BaseType),
    Pointer(Box<TypeKind>, AccessLevel),
    Custom(String),
    Wildcard(Wildcard),
}

impl TypeKind {
    pub fn base_type(&self) -> Option<&BaseType> {
        match self {
            Self::Vector(t) => Some(t.base_type()),
            Self::Pointer(t, _) => t.base_type(),
            Self::Base(t) => Some(t),
            Self::Wildcard(..) => None,
            Self::Custom(..) => None,
        }
    }

    pub fn base_type_mut(&mut self) -> Option<&mut BaseType> {
        match self {
            Self::Vector(t) => Some(t.base_type_mut()),
            Self::Pointer(t, _) => t.base_type_mut(),
            Self::Base(t) => Some(t),
            Self::Wildcard(..) => None,
            Self::Custom(..) => None,
        }
    }

    pub fn populate_wildcard(&mut self, type_kind: TypeKind) -> context::Result {
        match self {
            Self::Wildcard(..) => *self = type_kind,
            Self::Pointer(t, _) => t.populate_wildcard(type_kind)?,
            _ => return Err("no wildcard available to populate".to_string()),
        }
        Ok(())
    }

    pub fn base(&self) -> Option<&BaseType> {
        match self {
            Self::Base(ty) => Some(ty),
            Self::Pointer(tk, _) => tk.base(),
            Self::Vector(ty) => Some(&ty.base_type),
            _ => None,
        }
    }

    pub fn vector(&self) -> Option<&VectorType> {
        match self {
            Self::Vector(ty) => Some(ty),
            _ => None,
        }
    }

    pub fn vector_mut(&mut self) -> Option<&mut VectorType> {
        match self {
            Self::Vector(ty) => Some(ty),
            _ => None,
        }
    }

    pub fn wildcard(&self) -> Option<&Wildcard> {
        match self {
            Self::Wildcard(w) => Some(w),
            Self::Pointer(w, _) => w.wildcard(),
            _ => None,
        }
    }

    pub fn make_predicate_from(ty: &TypeKind) -> context::Result<TypeKind> {
        Ok(TypeKind::Vector(VectorType::make_predicate_from_bitsize(
            ty.base_type()
                .ok_or_else(|| format!("cannot infer predicate from type {ty}"))?
                .get_size()
                .map_err(|_| format!("cannot infer predicate from unsized type {ty}"))?,
        )))
    }

    pub fn make_vector(
        from: TypeKind,
        is_scalable: bool,
        tuple_size: Option<VectorTupleSize>,
    ) -> context::Result<TypeKind> {
        from.base().cloned().map_or_else(
            || Err(format!("cannot make a vector type out of {from}!")),
            |base| {
                let vt = VectorType::make_from_base(base, is_scalable, tuple_size);
                Ok(TypeKind::Vector(vt))
            },
        )
    }

    /// Return a new expression that converts the provided `expr` from type `other` to `self`.
    ///
    /// Conversions are bitwise over the whole value, like `transmute`, though `transmute`
    /// itself is only used as a last resort.
    ///
    /// This can fail (returning `None`) due to incompatible types, and many conversions are simply
    /// unimplemented.
    pub fn express_reinterpretation_from(
        &self,
        other: &TypeKind,
        expr: impl Into<Expression>,
    ) -> Option<Expression> {
        if self == other {
            Some(expr.into())
        } else if let (Some(self_vty), Some(other_vty)) = (self.vector(), other.vector()) {
            if self_vty.is_scalable
                && self_vty.tuple_size.is_none()
                && other_vty.is_scalable
                && other_vty.tuple_size.is_none()
            {
                // Plain scalable vectors.
                use BaseTypeKind::*;
                match (self_vty.base_type, other_vty.base_type) {
                    (BaseType::Sized(Int, self_size), BaseType::Sized(UInt, other_size))
                        if self_size == other_size =>
                    {
                        Some(Expression::MethodCall(
                            Box::new(expr.into()),
                            "as_signed".parse().unwrap(),
                            vec![],
                        ))
                    }
                    (BaseType::Sized(UInt, self_size), BaseType::Sized(Int, other_size))
                        if self_size == other_size =>
                    {
                        Some(Expression::MethodCall(
                            Box::new(expr.into()),
                            "as_unsigned".parse().unwrap(),
                            vec![],
                        ))
                    }
                    (
                        BaseType::Sized(Float | Int | UInt, _),
                        BaseType::Sized(Float | Int | UInt, _),
                    ) => Some(FnCall::new_expression(
                        // Conversions between float and (u)int, or where the lane size changes.
                        "simd_reinterpret".parse().unwrap(),
                        vec![expr.into()],
                    )),
                    _ => None,
                }
            } else {
                // Tuples and fixed-width vectors.
                None
            }
        } else {
            // Scalar types.
            None
        }
    }
}

impl FromStr for TypeKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            s if s.starts_with('{') && s.ends_with('}') => {
                Self::Wildcard(s[1..s.len() - 1].trim().parse()?)
            }
            s if s.starts_with('*') => {
                let mut split = s[1..].split_whitespace();
                let (ty, rw) = match (split.clone().count(), split.next(), split.next()) {
                    (2, Some("mut"), Some(ty)) => (ty, AccessLevel::RW),
                    (2, Some("const"), Some(ty)) => (ty, AccessLevel::R),
                    (1, Some(ty), None) => (ty, AccessLevel::R),
                    _ => return Err(format!("invalid pointer type {s:#?} given")),
                };
                Self::Pointer(Box::new(ty.parse()?), rw)
            }
            _ => s
                .parse::<VectorType>()
                .map(TypeKind::Vector)
                .or_else(|_| s.parse::<BaseType>().map(TypeKind::Base))
                .unwrap_or_else(|_| TypeKind::Custom(s.to_string())),
        })
    }
}

impl fmt::Display for TypeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vector(ty) => write!(f, "{ty}"),
            Self::Pointer(ty, _) => write!(f, "{ty}"),
            Self::Base(ty) => write!(f, "{ty}"),
            Self::Wildcard(w) => write!(f, "{{{w}}}"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

impl ToRepr for TypeKind {
    fn repr(&self, repr: TypeRepr) -> String {
        match self {
            Self::Vector(ty) => ty.repr(repr),
            Self::Pointer(ty, _) => ty.repr(repr),
            Self::Base(ty) => ty.repr(repr),
            Self::Wildcard(w) => format!("{w}"),
            Self::Custom(s) => s.to_string(),
        }
    }
}

impl ToTokens for TypeKind {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if let Self::Pointer(_, rw) = self {
            tokens.append_all(match rw {
                AccessLevel::RW => quote! { *mut },
                AccessLevel::R => quote! { *const },
            })
        }

        tokens.append_all(
            self.to_string()
                .parse::<TokenStream>()
                .expect("invalid syntax"),
        )
    }
}

impl PartialOrd for TypeKind {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl From<&TypeKind> for usize {
    fn from(ty: &TypeKind) -> Self {
        match ty {
            TypeKind::Base(_) => 1,
            TypeKind::Pointer(_, _) => 2,
            TypeKind::Vector(_) => 3,
            TypeKind::Custom(_) => 4,
            TypeKind::Wildcard(_) => 5,
        }
    }
}

impl Ord for TypeKind {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;

        let self_int: usize = self.into();
        let other_int: usize = other.into();

        if self_int == other_int {
            match (self, other) {
                (TypeKind::Base(ty1), TypeKind::Base(ty2)) => ty1.cmp(ty2),
                (TypeKind::Pointer(ty1, _), TypeKind::Pointer(ty2, _)) => ty1.cmp(ty2),
                (TypeKind::Vector(vt1), TypeKind::Vector(vt2)) => vt1.cmp(vt2),
                (TypeKind::Custom(s1), TypeKind::Custom(s2)) => s1.cmp(s2),
                (TypeKind::Wildcard(..), TypeKind::Wildcard(..)) => Equal,
                _ => unreachable!(),
            }
        } else {
            self_int.cmp(&other_int)
        }
    }
}

impl VectorType {
    pub fn base_type(&self) -> &BaseType {
        &self.base_type
    }

    pub fn base_type_mut(&mut self) -> &mut BaseType {
        &mut self.base_type
    }

    fn sanitise_lanes(
        mut base_type: BaseType,
        lanes: Option<u32>,
    ) -> Result<(BaseType, u32), String> {
        let lanes = match (base_type, lanes) {
            (BaseType::Sized(BaseTypeKind::Bool, lanes), None) => {
                base_type = BaseType::Sized(BaseTypeKind::Bool, VECTOR_FULL_REGISTER_SIZE / lanes);
                lanes
            }
            (BaseType::Unsized(BaseTypeKind::Bool), None) => {
                base_type = BaseType::Sized(BaseTypeKind::Bool, 8);
                16
            }
            (BaseType::Sized(_, size), None) => VECTOR_FULL_REGISTER_SIZE / size,
            (BaseType::Sized(_, size), Some(lanes)) => match size * lanes {
                VECTOR_FULL_REGISTER_SIZE | VECTOR_HALF_REGISTER_SIZE => lanes,
                _ => return Err("invalid number of lanes".to_string()),
            },
            _ => return Err("cannot infer number of lanes".to_string()),
        };

        Ok((base_type, lanes))
    }

    pub fn make_from_base(
        base_ty: BaseType,
        is_scalable: bool,
        tuple_size: Option<VectorTupleSize>,
    ) -> VectorType {
        #[allow(clippy::collapsible_if)]
        if is_scalable {
            if let BaseType::Sized(BaseTypeKind::Bool, size) = base_ty {
                return Self::make_predicate_from_bitsize(size);
            }
        }

        let (base_type, lanes) = Self::sanitise_lanes(base_ty, None).unwrap();

        VectorType {
            base_type,
            lanes,
            is_scalable,
            tuple_size,
        }
    }

    pub fn make_predicate_from_bitsize(size: u32) -> VectorType {
        VectorType {
            base_type: BaseType::Sized(BaseTypeKind::Bool, size),
            lanes: (VECTOR_FULL_REGISTER_SIZE / size),
            is_scalable: true,
            tuple_size: None,
        }
    }

    pub fn cast_base_type_as(&mut self, ty: BaseType) {
        self.base_type = ty
    }

    pub fn lanes(&self) -> u32 {
        self.lanes
    }

    pub fn tuple_size(&self) -> Option<VectorTupleSize> {
        self.tuple_size
    }
}

impl FromStr for VectorType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        static RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"^(?:(?:sv(?P<sv_ty>(?:uint|int|bool|float)(?:\d+)?))|(?:(?P<ty>(?:uint|int|bool|poly|float)(?:\d+)?)x(?P<lanes>(?:\d+)?)))(?:x(?P<tuple_size>2|3|4))?_t$").unwrap()
        });

        if let Some(c) = RE.captures(s) {
            let (base_type, lanes) = Self::sanitise_lanes(
                c.name("sv_ty")
                    .or_else(|| c.name("ty"))
                    .map(<&str>::from)
                    .map(BaseType::from_str)
                    .unwrap()?,
                c.name("lanes")
                    .map(<&str>::from)
                    .map(u32::from_str)
                    .transpose()
                    .unwrap(),
            )
            .map_err(|e| format!("invalid {s:#?} vector type: {e}"))?;

            let tuple_size = c
                .name("tuple_size")
                .map(<&str>::from)
                .map(VectorTupleSize::from_str)
                .transpose()
                .unwrap();

            Ok(VectorType {
                base_type,
                is_scalable: c.name("sv_ty").is_some(),
                lanes,
                tuple_size,
            })
        } else {
            Err(format!("invalid vector type {s:#?} given"))
        }
    }
}

impl ToRepr for VectorType {
    fn repr(&self, repr: TypeRepr) -> String {
        let make_llvm_repr = |show_unsigned| {
            format!(
                "{}v{}{}",
                if self.is_scalable { "nx" } else { "" },
                self.lanes * (self.tuple_size.map(usize::from).unwrap_or(1) as u32),
                match self.base_type {
                    BaseType::Sized(BaseTypeKind::UInt, size) if show_unsigned =>
                        format!("u{size}"),
                    _ => self.base_type.llvm_machine_repr(),
                }
            )
        };

        if matches!(repr, TypeRepr::ACLENotation) {
            self.base_type.acle_notation_repr()
        } else if matches!(repr, TypeRepr::LLVMMachine) {
            make_llvm_repr(false)
        } else if self.is_scalable {
            match (self.base_type, self.lanes, self.tuple_size) {
                (BaseType::Sized(BaseTypeKind::Bool, _), 16, _) => "svbool_t".to_string(),
                (BaseType::Sized(BaseTypeKind::Bool, _), lanes, _) => format!("svbool{lanes}_t"),
                (BaseType::Sized(_, size), lanes, _)
                    if VECTOR_FULL_REGISTER_SIZE != (size * lanes) =>
                {
                    // Special internal type case
                    make_llvm_repr(true)
                }
                (ty, _, None) => format!("sv{}_t", ty.c_repr()),
                (ty, _, Some(tuple_size)) => format!("sv{}x{tuple_size}_t", ty.c_repr()),
            }
        } else {
            match self.tuple_size {
                Some(tuple_size) => format!(
                    "{}x{}x{}_t",
                    self.base_type.c_repr(),
                    self.lanes,
                    tuple_size
                ),
                None => format!("{}x{}_t", self.base_type.c_repr(), self.lanes),
            }
        }
    }
}

impl fmt::Display for VectorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.c_repr())
    }
}

impl From<VectorTupleSize> for usize {
    fn from(t: VectorTupleSize) -> Self {
        match t {
            VectorTupleSize::Two => 2,
            VectorTupleSize::Three => 3,
            VectorTupleSize::Four => 4,
        }
    }
}

impl FromStr for VectorTupleSize {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "2" => Ok(Self::Two),
            "3" => Ok(Self::Three),
            "4" => Ok(Self::Four),
            _ => Err(format!("invalid vector tuple size `{s}` provided")),
        }
    }
}

impl TryFrom<usize> for VectorTupleSize {
    type Error = String;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            _ => Err(format!("invalid vector tuple size `{value}` provided")),
        }
    }
}

impl fmt::Display for VectorTupleSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", usize::from(*self))
    }
}

impl FromStr for BaseTypeKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "float" | "f" => Ok(Self::Float),
            "int" | "i" => Ok(Self::Int),
            "uint" | "u" => Ok(Self::UInt),
            "poly" | "p" => Ok(Self::Poly),
            "bool" | "b" => Ok(Self::Bool),
            _ => Err(format!("no match for {s}")),
        }
    }
}

impl ToRepr for BaseTypeKind {
    fn repr(&self, repr: TypeRepr) -> String {
        match (repr, self) {
            (TypeRepr::C, Self::Float) => "float",
            (TypeRepr::C, Self::Int) => "int",
            (TypeRepr::C, Self::UInt) => "uint",
            (TypeRepr::C, Self::Poly) => "poly",
            (TypeRepr::Rust | TypeRepr::LLVMMachine | TypeRepr::ACLENotation, Self::Float) => "f",
            (TypeRepr::Rust, Self::Int) | (TypeRepr::LLVMMachine, Self::Int | Self::UInt) => "i",
            (TypeRepr::Rust | TypeRepr::ACLENotation, Self::UInt) => "u",
            (TypeRepr::Rust | TypeRepr::LLVMMachine | TypeRepr::ACLENotation, Self::Poly) => "p",
            (TypeRepr::ACLENotation, Self::Int) => "s",
            (TypeRepr::ACLENotation, Self::Bool) => "b",
            (_, Self::Bool) => "bool",
            _ => {
                unreachable!("no base type kind available for representation {repr:?}")
            }
        }
        .to_string()
    }
}

impl fmt::Display for BaseTypeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.c_repr())
    }
}

impl BaseType {
    pub fn get_size(&self) -> Result<u32, String> {
        match self {
            Self::Sized(_, size) => Ok(*size),
            _ => Err(format!("unexpected invalid base type given {self:#?}")),
        }
    }

    pub fn kind(&self) -> &BaseTypeKind {
        match self {
            BaseType::Sized(kind, _) | BaseType::Unsized(kind) => kind,
        }
    }

    pub fn is_bool(&self) -> bool {
        self.kind() == &BaseTypeKind::Bool
    }

    pub fn is_float(&self) -> bool {
        self.kind() == &BaseTypeKind::Float
    }
}

impl FromStr for BaseType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        static RE: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"^(?P<kind>[a-zA-Z]+)(?P<size>\d+)?(_t)?$").unwrap());

        if let Some(c) = RE.captures(s) {
            let kind = c["kind"].parse()?;
            let size = c
                .name("size")
                .map(<&str>::from)
                .map(u32::from_str)
                .transpose()
                .unwrap();
            match size {
                Some(size) => Ok(Self::Sized(kind, size)),
                None => Ok(Self::Unsized(kind)),
            }
        } else {
            Err(format!("failed to parse type `{s}`"))
        }
    }
}

impl ToRepr for BaseType {
    fn repr(&self, repr: TypeRepr) -> String {
        use BaseType::*;
        use BaseTypeKind::*;
        use TypeRepr::*;
        match (self, &repr) {
            (Sized(Bool, _) | Unsized(Bool), LLVMMachine) => "i1".to_string(),
            (Sized(_, size), SizeLiteral) if *size == 8 => "b".to_string(),
            (Sized(_, size), SizeLiteral) if *size == 16 => "h".to_string(),
            (Sized(_, size), SizeLiteral) if *size == 32 => "w".to_string(),
            (Sized(_, size), SizeLiteral) if *size == 64 => "d".to_string(),
            (Sized(_, size), SizeLiteral) if *size == 128 => "q".to_string(),
            (_, SizeLiteral) => unreachable!("cannot represent {self:#?} as size literal"),
            (Sized(Float, _) | Unsized(Float), TypeKind) => "f".to_string(),
            (Sized(Int, _) | Unsized(Int), TypeKind) => "s".to_string(),
            (Sized(UInt, _) | Unsized(UInt), TypeKind) => "u".to_string(),
            (Sized(_, size), Size) => size.to_string(),
            (Sized(_, size), SizeInBytesLog2) => {
                assert!(size.is_power_of_two() && *size >= 8);
                (size >> 3).trailing_zeros().to_string()
            }
            (Sized(kind, size), _) => format!("{}{size}", kind.repr(repr)),
            (Unsized(kind), _) => kind.repr(repr),
        }
    }
}

impl fmt::Display for BaseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.rust_repr())
    }
}

#[cfg(test)]
mod tests {
    use crate::typekinds::*;

    #[test]
    fn test_predicate() {
        assert_eq!(
            "svbool_t".parse::<TypeKind>().unwrap(),
            TypeKind::Vector(VectorType {
                base_type: BaseType::Sized(BaseTypeKind::Bool, 8),
                is_scalable: true,
                lanes: 16,
                tuple_size: None
            })
        );
    }

    #[test]
    fn test_llvm_internal_predicate() {
        assert_eq!(
            "svbool4_t".parse::<TypeKind>().unwrap(),
            TypeKind::Vector(VectorType {
                base_type: BaseType::Sized(BaseTypeKind::Bool, 32),
                is_scalable: true,
                lanes: 4,
                tuple_size: None
            })
        );
    }

    #[test]
    fn test_llvm_internal_predicate_llvm() {
        assert_eq!(
            "svbool4_t".parse::<TypeKind>().unwrap().llvm_machine_repr(),
            "nxv4i1"
        );
    }

    #[test]
    fn test_llvm_internal_predicate_acle() {
        assert_eq!(
            "svbool4_t"
                .parse::<TypeKind>()
                .unwrap()
                .acle_notation_repr(),
            "b32"
        );
    }

    #[test]
    fn test_predicate_from_bitsize() {
        let pg = VectorType::make_predicate_from_bitsize(32);
        assert_eq!(pg.acle_notation_repr(), "b32");
        assert_eq!(pg, "svbool4_t".parse().unwrap());
        assert_eq!(pg.lanes, 4);
        assert_eq!(pg.base_type, BaseType::Sized(BaseTypeKind::Bool, 32));
    }

    #[test]
    fn test_scalable_single() {
        assert_eq!(
            "svuint8_t".parse::<TypeKind>().unwrap(),
            TypeKind::Vector(VectorType {
                base_type: BaseType::Sized(BaseTypeKind::UInt, 8),
                is_scalable: true,
                lanes: 16,
                tuple_size: None
            })
        );
    }

    #[test]
    fn test_scalable_tuple() {
        assert_eq!(
            "svint64x3_t".parse::<TypeKind>().unwrap(),
            TypeKind::Vector(VectorType {
                base_type: BaseType::Sized(BaseTypeKind::Int, 64),
                is_scalable: true,
                lanes: 2,
                tuple_size: Some(VectorTupleSize::Three),
            })
        );
    }

    #[test]
    fn test_scalable_single_llvm() {
        assert_eq!(
            "svuint32_t"
                .parse::<TypeKind>()
                .unwrap()
                .llvm_machine_repr(),
            "nxv4i32"
        );
    }

    #[test]
    fn test_scalable_tuple_llvm() {
        assert_eq!(
            "svint32x4_t"
                .parse::<TypeKind>()
                .unwrap()
                .llvm_machine_repr(),
            "nxv16i32"
        );
    }

    #[test]
    fn test_vector_single_full() {
        assert_eq!(
            "uint32x4_t".parse::<TypeKind>().unwrap(),
            TypeKind::Vector(VectorType {
                base_type: BaseType::Sized(BaseTypeKind::UInt, 32),
                is_scalable: false,
                lanes: 4,
                tuple_size: None,
            })
        );
    }

    #[test]
    fn test_vector_single_half() {
        assert_eq!(
            "uint32x2_t".parse::<TypeKind>().unwrap(),
            TypeKind::Vector(VectorType {
                base_type: BaseType::Sized(BaseTypeKind::UInt, 32),
                is_scalable: false,
                lanes: 2,
                tuple_size: None,
            })
        );
    }

    #[test]
    fn test_vector_tuple() {
        assert_eq!(
            "uint64x2x4_t".parse::<TypeKind>().unwrap(),
            TypeKind::Vector(VectorType {
                base_type: BaseType::Sized(BaseTypeKind::UInt, 64),
                is_scalable: false,
                lanes: 2,
                tuple_size: Some(VectorTupleSize::Four),
            })
        );
    }

    #[test]
    fn test_const_pointer() {
        let p = "*u32".parse::<TypeKind>().unwrap();
        assert_eq!(
            p,
            TypeKind::Pointer(
                Box::new(TypeKind::Base(BaseType::Sized(BaseTypeKind::UInt, 32))),
                AccessLevel::R
            )
        );
        assert_eq!(p.to_token_stream().to_string(), "* const u32")
    }

    #[test]
    fn test_mut_pointer() {
        let p = "*mut u32".parse::<TypeKind>().unwrap();
        assert_eq!(
            p,
            TypeKind::Pointer(
                Box::new(TypeKind::Base(BaseType::Sized(BaseTypeKind::UInt, 32))),
                AccessLevel::RW
            )
        );
        assert_eq!(p.to_token_stream().to_string(), "* mut u32")
    }

    #[test]
    #[should_panic]
    fn test_invalid_vector_single() {
        assert_eq!(
            "uint32x8_t".parse::<TypeKind>().unwrap(),
            TypeKind::Vector(VectorType {
                base_type: BaseType::Sized(BaseTypeKind::UInt, 32),
                is_scalable: false,
                lanes: 8,
                tuple_size: None,
            })
        );
    }

    #[test]
    #[should_panic]
    fn test_invalid_vector_tuple() {
        assert_eq!(
            "uint32x4x5_t".parse::<TypeKind>().unwrap(),
            TypeKind::Vector(VectorType {
                base_type: BaseType::Sized(BaseTypeKind::UInt, 32),
                is_scalable: false,
                lanes: 8,
                tuple_size: None, // cannot represent
            })
        );
    }

    #[test]
    fn test_base() {
        assert_eq!(
            "u32".parse::<TypeKind>().unwrap(),
            TypeKind::Base(BaseType::Sized(BaseTypeKind::UInt, 32)),
        )
    }

    #[test]
    fn test_custom() {
        assert_eq!(
            "svpattern".parse::<TypeKind>().unwrap(),
            TypeKind::Custom("svpattern".to_string()),
        )
    }

    #[test]
    fn test_wildcard_type() {
        assert_eq!(
            "{type}".parse::<TypeKind>().unwrap(),
            TypeKind::Wildcard(Wildcard::Type(None)),
        )
    }

    #[test]
    fn test_wildcard_typeset() {
        assert_eq!(
            "{type[0]}".parse::<TypeKind>().unwrap(),
            TypeKind::Wildcard(Wildcard::Type(Some(0))),
        )
    }

    #[test]
    fn test_wildcard_sve_type() {
        assert_eq!(
            "{sve_type}".parse::<TypeKind>().unwrap(),
            TypeKind::Wildcard(Wildcard::SVEType(None, None)),
        )
    }

    #[test]
    fn test_wildcard_sve_typeset() {
        assert_eq!(
            "{sve_type[0]}".parse::<TypeKind>().unwrap(),
            TypeKind::Wildcard(Wildcard::SVEType(Some(0), None)),
        )
    }

    #[test]
    fn test_wildcard_sve_tuple_type() {
        assert_eq!(
            "{sve_type_x2}".parse::<TypeKind>().unwrap(),
            TypeKind::Wildcard(Wildcard::SVEType(None, Some(VectorTupleSize::Two))),
        )
    }

    #[test]
    fn test_wildcard_sve_tuple_typeset() {
        assert_eq!(
            "{sve_type_x2[0]}".parse::<TypeKind>().unwrap(),
            TypeKind::Wildcard(Wildcard::SVEType(Some(0), Some(VectorTupleSize::Two))),
        )
    }

    #[test]
    fn test_wildcard_predicate() {
        assert_eq!(
            "{predicate}".parse::<TypeKind>().unwrap(),
            TypeKind::Wildcard(Wildcard::Predicate(None))
        )
    }

    #[test]
    fn test_wildcard_scale() {
        assert_eq!(
            "{sve_type as i8}".parse::<TypeKind>().unwrap(),
            TypeKind::Wildcard(Wildcard::Scale(
                Box::new(Wildcard::SVEType(None, None)),
                Box::new(TypeKind::Base(BaseType::Sized(BaseTypeKind::Int, 8)))
            ))
        )
    }

    #[test]
    fn test_size_in_bytes_log2() {
        assert_eq!("i8".parse::<TypeKind>().unwrap().size_in_bytes_log2(), "0");
        assert_eq!("i16".parse::<TypeKind>().unwrap().size_in_bytes_log2(), "1");
        assert_eq!("i32".parse::<TypeKind>().unwrap().size_in_bytes_log2(), "2");
        assert_eq!("i64".parse::<TypeKind>().unwrap().size_in_bytes_log2(), "3")
    }

    #[test]
    #[should_panic]
    fn test_invalid_size_in_bytes_log2() {
        "i9".parse::<TypeKind>().unwrap().size_in_bytes_log2();
    }
}
