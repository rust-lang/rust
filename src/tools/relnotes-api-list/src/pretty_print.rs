//! Convert the JSON representation of an `impl` signature into a textual representation of it,
//! closely aligning to Rust syntax.
//!
//! **WARNING:** changing how existing `impl` blocks are rendered will result in them showing up as
//! new items in the release notes. You should avoid making changes to this file, other than adding
//! support for new JSON types.

use anyhow::{Context, Error, anyhow};
use rustdoc_json_types::{
    Abi, AssocItemConstraintKind, Constant, FunctionHeader, FunctionSignature, GenericArg,
    GenericArgs, GenericBound, GenericParamDef, GenericParamDefKind, Impl, Path, PolyTrait,
    PreciseCapturingArg, Term, TraitBoundModifier, Type, WherePredicate,
};

use crate::store::{Store, StoreCrateId};
use crate::utils::urlencode;

pub(crate) struct PrettyPrinter<'a> {
    store: &'a Store,
    crate_id: StoreCrateId,

    hide_generic_params: bool,
    hide_negative_impls: bool,
    hide_lifetimes_in_references: bool,
    prefer_blanket_impl: bool,
}

impl<'a> PrettyPrinter<'a> {
    pub(crate) fn for_name(store: &'a Store, crate_id: StoreCrateId) -> Self {
        Self {
            store,
            crate_id,

            hide_generic_params: false,
            hide_negative_impls: false,
            hide_lifetimes_in_references: false,
            prefer_blanket_impl: false,
        }
    }

    pub(crate) fn for_url(store: &'a Store, crate_id: StoreCrateId) -> Self {
        Self {
            store,
            crate_id,

            hide_generic_params: true,
            hide_negative_impls: true,
            hide_lifetimes_in_references: true,
            prefer_blanket_impl: true,
        }
    }

    pub(crate) fn pretty_impl(&self, impl_: &Impl) -> Result<Chunked, Error> {
        // Each chunk is space-separated when emitting text, and `-`-separated when urlencoded.
        let mut chunked = Chunked::new();

        let mut prelude = "impl".to_string();
        if !impl_.generics.params.is_empty() && !self.hide_generic_params {
            prelude.push('<');
            prelude.push_str(
                &self.comma_separated(&impl_.generics.params, Self::pretty_generic_param_def)?,
            );
            prelude.push('>');
        }
        chunked.add(prelude);

        if let Some(trait_) = &impl_.trait_ {
            let mut path = String::new();
            if impl_.is_negative && !self.hide_negative_impls {
                path.push('!');
            }
            path.push_str(&self.pretty_path(&trait_)?);
            chunked.add(path);

            chunked.add("for");
        }
        if let Some(blanket) = &impl_.blanket_impl
            && self.prefer_blanket_impl
        {
            chunked.add(self.pretty_type(blanket)?);
        } else {
            chunked.add(self.pretty_type(&impl_.for_)?);
        }

        if !impl_.generics.where_predicates.is_empty() && !self.hide_generic_params {
            chunked.add("where");
            chunked.add(&self.comma_separated(
                &impl_.generics.where_predicates,
                Self::pretty_where_predicates,
            )?);
        }

        Ok(chunked)
    }

    fn pretty_path(&self, path: &Path) -> Result<String, Error> {
        let resolved = self
            .store
            .resolve_cross_crate(self.crate_id, path.id)
            .and_then(|resolved| self.store.item(resolved.krate, resolved.item))
            .with_context(|| format!("failed to resolve path {}", path.path))?;

        let mut result = resolved
            .name
            .clone()
            .ok_or_else(|| anyhow!("item the path {} points to does not have a name", path.path))?;
        if let Some(generic_args) = &path.args {
            result.push_str(&self.pretty_generic_args(generic_args)?);
        }
        Ok(result)
    }

    fn pretty_generic_args(&self, generic_args: &GenericArgs) -> Result<String, Error> {
        Ok(match generic_args {
            GenericArgs::AngleBracketed { args, constraints } => {
                let mut result = "<".to_string();
                for arg in args {
                    if result.len() > 1 {
                        result.push_str(", ");
                    }
                    result.push_str(&self.pretty_generic_arg(arg)?);
                }
                for constraint in constraints {
                    if result.len() > 1 {
                        result.push_str(", ");
                    }
                    result.push_str(&constraint.name);
                    if let Some(args) = &constraint.args {
                        result.push_str(&self.pretty_generic_args(args)?);
                    }
                    match &constraint.binding {
                        AssocItemConstraintKind::Equality(term) => {
                            result.push_str(" = ");
                            result.push_str(&self.pretty_term(term)?);
                        }
                        AssocItemConstraintKind::Constraint(bounds) => {
                            if !bounds.is_empty() {
                                result.push_str(": ");
                                result.push_str(
                                    &self.plus_separated(bounds, Self::pretty_generic_bound)?,
                                );
                            }
                        }
                    }
                }
                result.push('>');
                result
            }
            GenericArgs::Parenthesized { inputs, output } => {
                let mut result =
                    format!("({})", self.comma_separated(inputs.iter(), Self::pretty_type)?);
                if let Some(return_) = output {
                    result.push_str(" -> ");
                    result.push_str(&self.pretty_type(return_)?);
                }
                result
            }
            GenericArgs::ReturnTypeNotation => "(..)".to_string(),
        })
    }

    fn pretty_generic_arg(&self, generic_arg: &GenericArg) -> Result<String, Error> {
        Ok(match generic_arg {
            GenericArg::Lifetime(lifetime) => lifetime.clone(),
            GenericArg::Type(ty) => self.pretty_type(ty)?,
            GenericArg::Const(const_) => self.pretty_constant(const_)?,
            GenericArg::Infer => "_".into(),
        })
    }

    fn pretty_constant(&self, constant: &Constant) -> Result<String, Error> {
        Ok(if let Some(value) = &constant.value { value.clone() } else { constant.expr.clone() })
    }

    fn pretty_term(&self, term: &Term) -> Result<String, Error> {
        match term {
            Term::Type(type_) => self.pretty_type(&type_),
            Term::Constant(constant) => self.pretty_constant(&constant),
        }
    }

    fn pretty_type(&self, ty: &Type) -> Result<String, Error> {
        Ok(match ty {
            Type::ResolvedPath(path) => self.pretty_path(path)?,
            Type::DynTrait(dyn_trait) => {
                let mut result = format!(
                    "dyn {}",
                    self.plus_separated(dyn_trait.traits.iter(), Self::pretty_poly_trait)?
                );
                if let Some(lifetime) = &dyn_trait.lifetime {
                    result.push_str(lifetime);
                }
                result
            }
            Type::Generic(generic) => generic.clone(),
            Type::Primitive(primitive) => primitive.clone(),
            Type::FunctionPointer(function) => {
                format!(
                    "{}{}fn{}",
                    self.pretty_function_header(&function.header)?,
                    self.pretty_hrtb(&function.generic_params)?,
                    self.pretty_function_signature(&function.sig)?,
                )
            }
            Type::Tuple(inner) => {
                let trailing_comma = if inner.len() == 1 { "," } else { "" };
                format!("({}{trailing_comma})", self.comma_separated(inner, Self::pretty_type)?)
            }
            Type::Slice(inner) => format!("[{}]", self.pretty_type(inner)?),
            Type::Array { type_, len } => format!("[{}; {len}]", self.pretty_type(type_)?),
            Type::Pat { .. } => panic!("pattern type visible in the public api"),
            Type::ImplTrait(bounds) => {
                format!("impl {}", self.plus_separated(bounds, Self::pretty_generic_bound)?)
            }
            Type::Infer => "_".into(),
            Type::RawPointer { is_mutable, type_ } => {
                format!(
                    "*{} {}",
                    if *is_mutable { "mut" } else { "const" },
                    self.pretty_type(type_)?
                )
            }
            Type::BorrowedRef { lifetime, is_mutable, type_ } => {
                let lifetime = if self.hide_lifetimes_in_references { &None } else { lifetime };
                match (lifetime, is_mutable) {
                    (Some(lifetime), false) => format!("&{lifetime} {}", self.pretty_type(type_)?),
                    (Some(lifetime), true) => {
                        format!("&{lifetime} mut {}", self.pretty_type(type_)?)
                    }
                    (None, false) => format!("&{}", self.pretty_type(type_)?),
                    (None, true) => format!("&mut {}", self.pretty_type(type_)?),
                }
            }
            Type::QualifiedPath { name, args, self_type, trait_ } => {
                let mut result = format!("<{}", self.pretty_type(self_type)?);
                if let Some(trait_) = trait_ {
                    result.push_str(" as ");
                    result.push_str(&self.pretty_path(&trait_)?);
                }
                result.push_str(">::");
                result.push_str(name);
                if let Some(args) = args {
                    result.push_str(&self.pretty_generic_args(args)?);
                }
                result
            }
        })
    }

    fn pretty_poly_trait(&self, poly: &PolyTrait) -> Result<String, Error> {
        Ok(format!(
            "{}{}",
            self.pretty_hrtb(&poly.generic_params)?,
            self.pretty_path(&poly.trait_)?
        ))
    }

    fn pretty_hrtb(&self, gpds: &[GenericParamDef]) -> Result<String, Error> {
        let mut result = String::new();
        if !gpds.is_empty() {
            result.push_str("for<");
            result.push_str(&self.comma_separated(gpds, Self::pretty_generic_param_def)?);
            result.push_str("> ");
        }
        Ok(result)
    }

    fn pretty_generic_param_def(&self, gpd: &GenericParamDef) -> Result<Option<String>, Error> {
        Ok(Some(match &gpd.kind {
            GenericParamDefKind::Lifetime { outlives } => {
                if !outlives.is_empty() {
                    format!("{}: {}", gpd.name, self.plus_separated(outlives, Self::identity)?)
                } else {
                    gpd.name.clone()
                }
            }
            GenericParamDefKind::Type { bounds, default, is_synthetic } => {
                if *is_synthetic {
                    return Ok(None);
                }
                let mut result = gpd.name.clone();
                if !bounds.is_empty() {
                    result.push_str(": ");
                    result.push_str(&self.plus_separated(bounds, Self::pretty_generic_bound)?);
                }
                if let Some(default) = default {
                    result.push_str(" = ");
                    result.push_str(&self.pretty_type(default)?);
                }
                result
            }
            GenericParamDefKind::Const { type_, default } => {
                let mut result = format!("const {}: {}", gpd.name, self.pretty_type(type_)?);
                if let Some(default) = default {
                    result.push_str(" = ");
                    result.push_str(default);
                }
                result
            }
        }))
    }

    fn pretty_generic_bound(&self, bound: &GenericBound) -> Result<String, Error> {
        Ok(match bound {
            GenericBound::TraitBound { trait_, generic_params, modifier } => {
                let mut result =
                    format!("{}{}", self.pretty_hrtb(generic_params)?, self.pretty_path(trait_)?);
                match modifier {
                    TraitBoundModifier::None => {}
                    TraitBoundModifier::Maybe => result.push('?'),
                    TraitBoundModifier::MaybeConst => {}
                }
                result
            }
            GenericBound::Outlives(lifetime) => format!("{lifetime}"),
            GenericBound::Use(use_) => {
                format!(
                    "use<{}>",
                    self.comma_separated(use_, |_self, arg| Ok(match arg {
                        PreciseCapturingArg::Lifetime(lifetime) => lifetime,
                        PreciseCapturingArg::Param(param) => param,
                    }))?
                )
            }
        })
    }

    fn pretty_where_predicates(&self, predicate: &WherePredicate) -> Result<String, Error> {
        Ok(match predicate {
            WherePredicate::BoundPredicate { type_, bounds, generic_params } => {
                format!(
                    "{}{}: {}",
                    self.pretty_hrtb(generic_params)?,
                    self.pretty_type(type_)?,
                    self.plus_separated(bounds, Self::pretty_generic_bound)?,
                )
            }
            WherePredicate::LifetimePredicate { lifetime, outlives } => {
                format!("{lifetime}: {}", self.plus_separated(outlives, Self::identity)?)
            }
            WherePredicate::EqPredicate { lhs, rhs } => {
                format!("{} = {}", self.pretty_type(lhs)?, self.pretty_term(rhs)?)
            }
        })
    }

    fn pretty_function_header(&self, header: &FunctionHeader) -> Result<String, Error> {
        let mut result = String::new();
        if header.is_const {
            result.push_str("const ");
        }
        if header.is_unsafe {
            result.push_str("unsafe ");
        }
        if header.is_async {
            result.push_str("async ");
        }

        let mut abi = |name, unwind: &bool| {
            result.push_str(&if *unwind {
                format!("extern \"{name}-unwind\" ")
            } else {
                format!("extern \"{name}\" ")
            })
        };
        match &header.abi {
            Abi::Rust => {}
            Abi::C { unwind } => abi("C", unwind),
            Abi::Cdecl { unwind } => abi("Cdecl", unwind),
            Abi::Stdcall { unwind } => abi("stdcall", unwind),
            Abi::Fastcall { unwind } => abi("fastcall", unwind),
            Abi::Aapcs { unwind } => abi("aapcs", unwind),
            Abi::Win64 { unwind } => abi("win64", unwind),
            Abi::SysV64 { unwind } => abi("sysv", unwind),
            Abi::System { unwind } => abi("system", unwind),
            Abi::Other(name) => abi(&name, &false),
        };

        Ok(result)
    }

    fn pretty_function_signature(&self, sig: &FunctionSignature) -> Result<String, Error> {
        let mut result = format!(
            "({}",
            self.comma_separated(&sig.inputs, |_self, (name, ty)| Ok(format!(
                "{name}: {}",
                self.pretty_type(ty)?
            )))?
        );
        if sig.is_c_variadic {
            if result.len() > 1 {
                result.push_str(", ");
            }
            result.push_str("..")
        }
        result.push(')');
        if let Some(output) = &sig.output {
            result.push_str(" -> ");
            result.push_str(&self.pretty_type(&output)?);
        }
        Ok(result)
    }

    fn comma_separated<T, I, F, R>(&self, iter: I, func: F) -> Result<String, Error>
    where
        I: IntoIterator<Item = T>,
        R: SeparatedInput,
        F: Fn(&Self, T) -> Result<R, Error>,
    {
        self.separated(", ", iter, func)
    }

    fn plus_separated<T, I, F, R>(&self, iter: I, func: F) -> Result<String, Error>
    where
        I: IntoIterator<Item = T>,
        R: SeparatedInput,
        F: Fn(&Self, T) -> Result<R, Error>,
    {
        self.separated(" + ", iter, func)
    }

    fn separated<T, I, F, R>(&self, separator: &str, iter: I, func: F) -> Result<String, Error>
    where
        I: IntoIterator<Item = T>,
        R: SeparatedInput,
        F: Fn(&Self, T) -> Result<R, Error>,
    {
        let mut result = String::new();
        let mut first = true;
        for item in iter {
            let item_result = func(self, item)?;
            let Some(item) = item_result.into_option_str() else { continue };
            if first {
                first = false;
            } else {
                result.push_str(separator);
            }
            result.push_str(item);
        }
        Ok(result)
    }

    fn identity<T>(&self, value: T) -> Result<T, Error> {
        Ok(value)
    }
}

pub(crate) struct Chunked {
    chunks: Vec<String>,
}

impl Chunked {
    fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    fn add(&mut self, chunk: impl Into<String>) {
        self.chunks.push(chunk.into());
    }

    pub(crate) fn as_text(&self) -> String {
        self.chunks.join(" ")
    }

    pub(crate) fn as_urlencoded(&self) -> String {
        // When generating anchors for `impl`s, rustdoc separates each "chunk" of the anchor (the
        // type name, trait name, `where`, etc) with a `-` (effectively replacing each space with
        // `-`), while inside of each chunk the content is urlencoded (replacing spaces with `+`).
        self.chunks.iter().map(|chunk| urlencode(&chunk)).collect::<Vec<_>>().join("-")
    }
}

trait SeparatedInput {
    fn into_option_str(&self) -> Option<&str>;
}

impl SeparatedInput for &str {
    fn into_option_str(&self) -> Option<&str> {
        Some(self)
    }
}

impl SeparatedInput for String {
    fn into_option_str(&self) -> Option<&str> {
        Some(self.as_str())
    }
}

impl<T: SeparatedInput> SeparatedInput for Option<T> {
    fn into_option_str(&self) -> Option<&str> {
        self.as_ref().and_then(|s| s.into_option_str())
    }
}

impl<T: SeparatedInput> SeparatedInput for &T {
    fn into_option_str(&self) -> Option<&str> {
        (*self).into_option_str()
    }
}
