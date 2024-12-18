use lazy_static::lazy_static;
use regex::Regex;
use serde_with::{DeserializeFromStr, SerializeDisplay};
use std::fmt;
use std::str::FromStr;

use crate::{
    fn_suffix::SuffixKind,
    predicate_forms::PredicationMask,
    typekinds::{ToRepr, TypeKind, TypeKindOptions, VectorTupleSize},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, SerializeDisplay, DeserializeFromStr)]
pub enum Wildcard {
    Type(Option<usize>),
    /// NEON type derivated by a base type
    NEONType(Option<usize>, Option<VectorTupleSize>, Option<SuffixKind>),
    /// SVE type derivated by a base type
    SVEType(Option<usize>, Option<VectorTupleSize>),
    /// Integer representation of bitsize
    Size(Option<usize>),
    /// Integer representation of bitsize minus one
    SizeMinusOne(Option<usize>),
    /// Literal representation of the bitsize: b(yte), h(half), w(ord) or d(ouble)
    SizeLiteral(Option<usize>),
    /// Literal representation of the type kind: f(loat), s(igned), u(nsigned)
    TypeKind(Option<usize>, Option<TypeKindOptions>),
    /// Log2 of the size in bytes
    SizeInBytesLog2(Option<usize>),
    /// Predicate to be inferred from the specified type
    Predicate(Option<usize>),
    /// Predicate to be inferred from the greatest type
    MaxPredicate,

    Scale(Box<Wildcard>, Box<TypeKind>),

    // Other wildcards
    LLVMLink,
    NVariant,
    /// Predicate forms to use and placeholder for a predicate form function name modifier
    PredicateForms(PredicationMask),

    /// User-set wildcard through `substitutions`
    Custom(String),
}

impl Wildcard {
    pub fn is_nonpredicate_type(&self) -> bool {
        matches!(
            self,
            Wildcard::Type(..) | Wildcard::NEONType(..) | Wildcard::SVEType(..)
        )
    }

    pub fn get_typeset_index(&self) -> Option<usize> {
        match self {
            Wildcard::Type(idx) | Wildcard::NEONType(idx, ..) | Wildcard::SVEType(idx, ..) => {
                Some(idx.unwrap_or(0))
            }
            _ => None,
        }
    }
}

impl FromStr for Wildcard {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        lazy_static! {
            static ref RE: Regex = Regex::new(r"^(?P<wildcard>\w+?)(?:_x(?P<tuple_size>[2-4]))?(?:\[(?P<index>\d+)\])?(?:\.(?P<modifiers>\w+))?(?:\s+as\s+(?P<scale_to>.*?))?$").unwrap();
        }

        if let Some(c) = RE.captures(s) {
            let wildcard_name = &c["wildcard"];
            let inputset_index = c
                .name("index")
                .map(<&str>::from)
                .map(usize::from_str)
                .transpose()
                .map_err(|_| format!("{:#?} is not a valid type index", &c["index"]))?;
            let tuple_size = c
                .name("tuple_size")
                .map(<&str>::from)
                .map(VectorTupleSize::from_str)
                .transpose()
                .map_err(|_| format!("{:#?} is not a valid tuple size", &c["tuple_size"]))?;
            let modifiers = c.name("modifiers").map(<&str>::from);

            let wildcard = match (wildcard_name, inputset_index, tuple_size, modifiers) {
                ("type", index, None, None) => Ok(Wildcard::Type(index)),
                ("neon_type", index, tuple, modifier) => {
                    if let Some(str_suffix) = modifier {
                        let suffix_kind = SuffixKind::from_str(str_suffix);
                        return Ok(Wildcard::NEONType(index, tuple, Some(suffix_kind.unwrap())));
                    } else {
                        Ok(Wildcard::NEONType(index, tuple, None))
                    }
                }
                ("sve_type", index, tuple, None) => Ok(Wildcard::SVEType(index, tuple)),
                ("size", index, None, None) => Ok(Wildcard::Size(index)),
                ("size_minus_one", index, None, None) => Ok(Wildcard::SizeMinusOne(index)),
                ("size_literal", index, None, None) => Ok(Wildcard::SizeLiteral(index)),
                ("type_kind", index, None, modifiers) => Ok(Wildcard::TypeKind(
                    index,
                    modifiers.map(|modifiers| modifiers.parse()).transpose()?,
                )),
                ("size_in_bytes_log2", index, None, None) => Ok(Wildcard::SizeInBytesLog2(index)),
                ("predicate", index, None, None) => Ok(Wildcard::Predicate(index)),
                ("max_predicate", None, None, None) => Ok(Wildcard::MaxPredicate),
                ("llvm_link", None, None, None) => Ok(Wildcard::LLVMLink),
                ("_n", None, None, None) => Ok(Wildcard::NVariant),
                (w, None, None, None) if w.starts_with('_') => {
                    // test for predicate forms
                    let pf_mask = PredicationMask::from_str(&w[1..]);
                    if let Ok(mask) = pf_mask {
                        if mask.has_merging() {
                            Ok(Wildcard::PredicateForms(mask))
                        } else {
                            Err("cannot add predication without a Merging form".to_string())
                        }
                    } else {
                        Err(format!("invalid wildcard `{s:#?}`"))
                    }
                }
                (cw, None, None, None) => Ok(Wildcard::Custom(cw.to_string())),
                _ => Err(format!("invalid wildcard `{s:#?}`")),
            }?;

            let scale_to = c
                .name("scale_to")
                .map(<&str>::from)
                .map(TypeKind::from_str)
                .transpose()
                .map_err(|_| format!("{:#?} is not a valid type", &c["scale_to"]))?;

            if let Some(scale_to) = scale_to {
                Ok(Wildcard::Scale(Box::new(wildcard), Box::new(scale_to)))
            } else {
                Ok(wildcard)
            }
        } else {
            Err(format!("## invalid wildcard `{s:#?}`"))
        }
    }
}

impl fmt::Display for Wildcard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Type(None) => write!(f, "type"),
            Self::Type(Some(index)) => write!(f, "type[{index}]"),
            Self::NEONType(None, None, None) => write!(f, "neon_type"),
            Self::NEONType(None, None, Some(suffix_kind)) => write!(f, "neon_type.{suffix_kind}"),
            Self::NEONType(Some(index), None, None) => write!(f, "neon_type[{index}]"),
            Self::NEONType(Some(index), None, Some(suffix_kind)) => {
                write!(f, "neon_type[{index}].{suffix_kind}")
            }
            Self::NEONType(None, Some(tuple_size), Some(suffix_kind)) => {
                write!(f, "neon_type_x{tuple_size}.{suffix_kind}")
            }
            Self::NEONType(None, Some(tuple_size), None) => write!(f, "neon_type_x{tuple_size}"),
            Self::NEONType(Some(index), Some(tuple_size), None) => {
                write!(f, "neon_type_x{tuple_size}[{index}]")
            }
            Self::NEONType(Some(index), Some(tuple_size), Some(suffix_kind)) => {
                write!(f, "neon_type_x{tuple_size}[{index}].{suffix_kind}")
            }
            Self::SVEType(None, None) => write!(f, "sve_type"),
            Self::SVEType(Some(index), None) => write!(f, "sve_type[{index}]"),
            Self::SVEType(None, Some(tuple_size)) => write!(f, "sve_type_x{tuple_size}"),
            Self::SVEType(Some(index), Some(tuple_size)) => {
                write!(f, "sve_type_x{tuple_size}[{index}]")
            }
            Self::Size(None) => write!(f, "size"),
            Self::Size(Some(index)) => write!(f, "size[{index}]"),
            Self::SizeMinusOne(None) => write!(f, "size_minus_one"),
            Self::SizeMinusOne(Some(index)) => write!(f, "size_minus_one[{index}]"),
            Self::SizeLiteral(None) => write!(f, "size_literal"),
            Self::SizeLiteral(Some(index)) => write!(f, "size_literal[{index}]"),
            Self::TypeKind(None, None) => write!(f, "type_kind"),
            Self::TypeKind(None, Some(opts)) => write!(f, "type_kind.{opts}"),
            Self::TypeKind(Some(index), None) => write!(f, "type_kind[{index}]"),
            Self::TypeKind(Some(index), Some(opts)) => write!(f, "type_kind[{index}].{opts}"),
            Self::SizeInBytesLog2(None) => write!(f, "size_in_bytes_log2"),
            Self::SizeInBytesLog2(Some(index)) => write!(f, "size_in_bytes_log2[{index}]"),
            Self::Predicate(None) => write!(f, "predicate"),
            Self::Predicate(Some(index)) => write!(f, "predicate[{index}]"),
            Self::MaxPredicate => write!(f, "max_predicate"),
            Self::LLVMLink => write!(f, "llvm_link"),
            Self::NVariant => write!(f, "_n"),
            Self::PredicateForms(mask) => write!(f, "_{mask}"),

            Self::Scale(wildcard, ty) => write!(f, "{wildcard} as {}", ty.rust_repr()),
            Self::Custom(cw) => write!(f, "{cw}"),
        }
    }
}
