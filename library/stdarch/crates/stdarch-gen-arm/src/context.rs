use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, usize};

use crate::{
    expression::Expression,
    input::{InputSet, InputType},
    intrinsic::{Constraint, Intrinsic, Signature},
    matching::SizeMatchable,
    predicate_forms::PredicateForm,
    typekinds::{ToRepr, TypeKind},
    wildcards::Wildcard,
    wildstring::WildString,
};

/// Maximum SVE vector size
const SVE_VECTOR_MAX_SIZE: u32 = 2048;
/// Vector register size
const VECTOR_REG_SIZE: u32 = 128;

/// Generator result
pub type Result<T = ()> = std::result::Result<T, String>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureSettings {
    #[serde(alias = "arch")]
    pub arch_name: String,
    pub target_feature: Vec<String>,
    #[serde(alias = "llvm_prefix")]
    pub llvm_link_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalContext {
    pub arch_cfgs: Vec<ArchitectureSettings>,
    #[serde(default)]
    pub uses_neon_types: bool,

    /// Should the yaml file automagically generate big endian shuffling
    #[serde(default)]
    pub auto_big_endian: Option<bool>,

    /// Should all LLVM wrappers convert their arguments to a signed type
    #[serde(default)]
    pub auto_llvm_sign_conversion: bool,
}

/// Context of an intrinsic group
#[derive(Debug, Clone, Default)]
pub struct GroupContext {
    /// LLVM links to target input sets
    pub links: HashMap<String, InputSet>,
}

#[derive(Debug, Clone, Copy)]
pub enum VariableType {
    Argument,
    Internal,
}

#[derive(Debug, Clone)]
pub struct LocalContext {
    pub signature: Signature,

    pub input: InputSet,

    pub substitutions: HashMap<Wildcard, String>,
    pub variables: HashMap<String, (TypeKind, VariableType)>,
}

impl LocalContext {
    pub fn new(input: InputSet, original: &Intrinsic) -> LocalContext {
        LocalContext {
            signature: original.signature.clone(),
            input,
            substitutions: HashMap::new(),
            variables: HashMap::new(),
        }
    }

    pub fn provide_type_wildcard(&self, wildcard: &Wildcard) -> Result<TypeKind> {
        let err = || {
            format!(
                "provide_type_wildcard() wildcard {{{wildcard}}} not found for {}",
                &self.signature.name.to_string()
            )
        };

        /* If the type is already a vector then we can just return the vector */
        let make_neon = |tuple_size| {
            move |ty| match ty {
                TypeKind::Vector(_) => Ok(ty),
                _ => TypeKind::make_vector(ty, false, tuple_size),
            }
        };
        let make_sve = |tuple_size| move |ty| TypeKind::make_vector(ty, true, tuple_size);

        match wildcard {
            Wildcard::Type(idx) => self.input.typekind(*idx).ok_or_else(err),
            Wildcard::NEONType(idx, tuple_size, _) => self
                .input
                .typekind(*idx)
                .ok_or_else(|| {
                    dbg!("{:?}", &self);
                    err()
                })
                .and_then(make_neon(*tuple_size)),
            Wildcard::SVEType(idx, tuple_size) => self
                .input
                .typekind(*idx)
                .ok_or_else(err)
                .and_then(make_sve(*tuple_size)),
            Wildcard::Predicate(idx) => self.input.typekind(*idx).map_or_else(
                || {
                    if idx.is_none() && self.input.types_len() == 1 {
                        Err(err())
                    } else {
                        Err(format!(
                            "there is no type at index {} to infer the predicate from",
                            idx.unwrap_or(0)
                        ))
                    }
                },
                |ref ty| TypeKind::make_predicate_from(ty),
            ),
            Wildcard::MaxPredicate => self
                .input
                .iter()
                .filter_map(|arg| arg.typekind())
                .max_by(|x, y| {
                    x.base_type()
                        .and_then(|bt| bt.get_size().ok())
                        .unwrap_or(0)
                        .cmp(&y.base_type().and_then(|bt| bt.get_size().ok()).unwrap_or(0))
                })
                .map_or_else(
                    || Err("there are no types available to infer the predicate from".to_string()),
                    TypeKind::make_predicate_from,
                ),
            Wildcard::Scale(w, as_ty) => {
                let mut ty = self.provide_type_wildcard(w)?;
                if let Some(vty) = ty.vector_mut() {
                    let base_ty = if let Some(w) = as_ty.wildcard() {
                        *self.provide_type_wildcard(w)?.base_type().unwrap()
                    } else {
                        *as_ty.base_type().unwrap()
                    };
                    vty.cast_base_type_as(base_ty)
                }
                Ok(ty)
            }
            _ => Err(err()),
        }
    }

    pub fn provide_substitution_wildcard(&self, wildcard: &Wildcard) -> Result<String> {
        let err = || Err(format!("wildcard {{{wildcard}}} not found"));

        match wildcard {
            Wildcard::SizeLiteral(idx) => self.input.typekind(*idx)
                .map_or_else(err, |ty| Ok(ty.size_literal())),
            Wildcard::Size(idx) => self.input.typekind(*idx)
                .map_or_else(err, |ty| Ok(ty.size())),
            Wildcard::SizeMinusOne(idx) => self.input.typekind(*idx)
                .map_or_else(err, |ty| Ok((ty.size().parse::<i32>().unwrap()-1).to_string())),
            Wildcard::SizeInBytesLog2(idx) => self.input.typekind(*idx)
                .map_or_else(err, |ty| Ok(ty.size_in_bytes_log2())),
            Wildcard::NVariant if self.substitutions.get(wildcard).is_none() => Ok(String::new()),
            Wildcard::TypeKind(idx, opts) => {
                self.input.typekind(*idx)
                    .map_or_else(err, |ty| {
                        let literal = if let Some(opts) = opts {
                            opts.contains(ty.base_type().map(|bt| *bt.kind()).ok_or_else(|| {
                                format!("cannot retrieve a type literal out of {ty}")
                            })?)
                            .then(|| ty.type_kind())
                            .unwrap_or_default()
                        } else {
                            ty.type_kind()
                        };
                        Ok(literal)
                    })
            }
            Wildcard::PredicateForms(_) => self
                .input
                .iter()
                .find_map(|arg| {
                    if let InputType::PredicateForm(pf) = arg {
                        Some(pf.get_suffix().to_string())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| unreachable!("attempting to render a predicate form wildcard, but no predicate form was compiled for it")),
            _ => self
                .substitutions
                .get(wildcard)
                .map_or_else(err, |s| Ok(s.clone())),
        }
    }

    pub fn make_assertion_from_constraint(&self, constraint: &Constraint) -> Result<Expression> {
        match constraint {
            Constraint::AnyI32 {
                variable,
                any_values,
            } => {
                let where_ex = any_values
                    .iter()
                    .map(|value| format!("{variable} == {value}"))
                    .join(" || ");
                Ok(Expression::MacroCall("static_assert".to_string(), where_ex))
            }
            Constraint::RangeI32 {
                variable,
                range: SizeMatchable::Matched(range),
            } => Ok(Expression::MacroCall(
                "static_assert_range".to_string(),
                format!(
                    "{variable}, {min}, {max}",
                    min = range.start(),
                    max = range.end()
                ),
            )),
            Constraint::SVEMaxElems {
                variable,
                sve_max_elems_type: ty,
            }
            | Constraint::VecMaxElems {
                variable,
                vec_max_elems_type: ty,
            } => {
                if !self.input.is_empty() {
                    let higher_limit = match constraint {
                        Constraint::SVEMaxElems { .. } => SVE_VECTOR_MAX_SIZE,
                        Constraint::VecMaxElems { .. } => VECTOR_REG_SIZE,
                        _ => unreachable!(),
                    };

                    let max = ty.base_type()
                        .map(|ty| ty.get_size())
                        .transpose()?
                        .map_or_else(
                            || Err(format!("can't make an assertion out of constraint {self:?}: no valid type is present")),
                            |bitsize| Ok(higher_limit / bitsize - 1))?;
                    Ok(Expression::MacroCall(
                        "static_assert_range".to_string(),
                        format!("{variable}, 0, {max}"),
                    ))
                } else {
                    Err(format!(
                        "can't make an assertion out of constraint {self:?}: no types are being used"
                    ))
                }
            }
            _ => unreachable!("constraints were not built successfully!"),
        }
    }

    pub fn predicate_form(&self) -> Option<&PredicateForm> {
        self.input.iter().find_map(|arg| arg.predicate_form())
    }

    pub fn n_variant_op(&self) -> Option<&WildString> {
        self.input.iter().find_map(|arg| arg.n_variant_op())
    }
}

pub struct Context<'ctx> {
    pub local: &'ctx mut LocalContext,
    pub group: &'ctx mut GroupContext,
    pub global: &'ctx GlobalContext,
}
