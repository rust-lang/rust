//! Convert the JSON representation of an `impl` signature into a textual representation of it,
//! closely aligning to Rust syntax.
//!
//! **WARNING:** changing how existing `impl` blocks are rendered will result in them showing up as
//! new items in the release notes. You should avoid making changes to this file, other than adding
//! support for new JSON types.

use std::convert::identity;

use rustdoc_json_types::{
    Abi, AssocItemConstraintKind, Constant, FunctionHeader, FunctionSignature, GenericArg,
    GenericArgs, GenericBound, GenericParamDef, GenericParamDefKind, Impl, Path, PolyTrait,
    PreciseCapturingArg, Term, TraitBoundModifier, Type, WherePredicate,
};

pub(crate) fn pretty_impl(impl_: &Impl) -> String {
    let mut result = "impl".to_string();

    if !impl_.generics.params.is_empty() {
        result.push('<');
        result.push_str(&comma_separated(&impl_.generics.params, pretty_generic_param_def));
        result.push('>');
    }

    result.push(' ');
    if let Some(trait_) = &impl_.trait_ {
        if impl_.is_negative {
            result.push('!');
        }
        result.push_str(&pretty_path(&trait_));
        result.push_str(" for ");
    }
    result.push_str(&pretty_type(&impl_.for_));

    if !impl_.generics.where_predicates.is_empty() {
        result.push_str(" where ");
        result
            .push_str(&comma_separated(&impl_.generics.where_predicates, pretty_where_predicates));
    }

    result
}

pub(crate) fn pretty_path(path: &Path) -> String {
    let mut result = path.path.clone();
    if let Some(generic_args) = &path.args {
        result.push_str(&pretty_generic_args(generic_args));
    }
    result
}

fn pretty_generic_args(generic_args: &GenericArgs) -> String {
    match generic_args {
        GenericArgs::AngleBracketed { args, constraints } => {
            let mut result = "<".to_string();
            for arg in args {
                if result.len() > 1 {
                    result.push_str(", ");
                }
                result.push_str(&pretty_generic_arg(arg));
            }
            for constraint in constraints {
                if result.len() > 1 {
                    result.push_str(", ");
                }
                result.push_str(&constraint.name);
                if let Some(args) = &constraint.args {
                    result.push_str(&pretty_generic_args(args));
                }
                match &constraint.binding {
                    AssocItemConstraintKind::Equality(term) => {
                        result.push_str(" = ");
                        result.push_str(&pretty_term(term));
                    }
                    AssocItemConstraintKind::Constraint(bounds) => {
                        if !bounds.is_empty() {
                            result.push_str(": ");
                            result.push_str(&plus_separated(bounds, pretty_generic_bound));
                        }
                    }
                }
            }
            result.push('>');
            result
        }
        GenericArgs::Parenthesized { inputs, output } => {
            let mut result = format!("({})", comma_separated(inputs.iter(), pretty_type));
            if let Some(return_) = output {
                result.push_str(" -> ");
                result.push_str(&pretty_type(return_));
            }
            result
        }
        GenericArgs::ReturnTypeNotation => "(..)".to_string(),
    }
}

fn pretty_generic_arg(generic_arg: &GenericArg) -> String {
    match generic_arg {
        GenericArg::Lifetime(lifetime) => lifetime.clone(),
        GenericArg::Type(ty) => pretty_type(ty),
        GenericArg::Const(const_) => pretty_constant(const_),
        GenericArg::Infer => "_".into(),
    }
}

fn pretty_constant(constant: &Constant) -> String {
    if let Some(value) = &constant.value { value.clone() } else { constant.expr.clone() }
}

fn pretty_term(term: &Term) -> String {
    match term {
        Term::Type(type_) => pretty_type(&type_),
        Term::Constant(constant) => pretty_constant(&constant),
    }
}

fn pretty_type(ty: &Type) -> String {
    match ty {
        Type::ResolvedPath(path) => pretty_path(path),
        Type::DynTrait(dyn_trait) => {
            let mut result =
                format!("dyn {}", plus_separated(dyn_trait.traits.iter(), pretty_poly_trait));
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
                pretty_function_header(&function.header),
                pretty_hrtb(&function.generic_params),
                pretty_function_signature(&function.sig),
            )
        }
        Type::Tuple(inner) => format!("({})", comma_separated(inner, pretty_type)),
        Type::Slice(inner) => format!("[{}]", pretty_type(inner)),
        Type::Array { type_, len } => format!("[{}; {len}]", pretty_type(type_)),
        Type::Pat { .. } => panic!("pattern type visible in the public api"),
        Type::ImplTrait(bounds) => {
            format!("impl {}", plus_separated(bounds, pretty_generic_bound))
        }
        Type::Infer => "_".into(),
        Type::RawPointer { is_mutable, type_ } => {
            format!("*{} {}", if *is_mutable { "mut" } else { "const" }, pretty_type(type_))
        }
        Type::BorrowedRef { lifetime, is_mutable, type_ } => match (lifetime, is_mutable) {
            (Some(lifetime), false) => format!("&{lifetime} {}", pretty_type(type_)),
            (Some(lifetime), true) => format!("&{lifetime} mut {}", pretty_type(type_)),
            (None, false) => format!("&{}", pretty_type(type_)),
            (None, true) => format!("&mut {}", pretty_type(type_)),
        },
        Type::QualifiedPath { name, args, self_type, trait_ } => {
            let mut result = format!("<{}", pretty_type(self_type));
            if let Some(trait_) = trait_ {
                result.push_str(" as ");
                result.push_str(&pretty_path(&trait_));
            }
            result.push_str(">::");
            result.push_str(name);
            if let Some(args) = args {
                result.push_str(&pretty_generic_args(args));
            }
            result
        }
    }
}

fn pretty_poly_trait(poly: &PolyTrait) -> String {
    format!("{}{}", pretty_hrtb(&poly.generic_params), pretty_path(&poly.trait_))
}

fn pretty_hrtb(gpds: &[GenericParamDef]) -> String {
    let mut result = String::new();
    if !gpds.is_empty() {
        result.push_str("for<");
        comma_separated(gpds, pretty_generic_param_def);
        result.push_str("> ");
    }
    result
}

fn pretty_generic_param_def(gpd: &GenericParamDef) -> Option<String> {
    match &gpd.kind {
        GenericParamDefKind::Lifetime { outlives } => {
            if !outlives.is_empty() {
                Some(format!("{}: {}", gpd.name, plus_separated(outlives, identity)))
            } else {
                Some(gpd.name.clone())
            }
        }
        GenericParamDefKind::Type { bounds, default, is_synthetic } => {
            if *is_synthetic {
                return None;
            }
            let mut result = gpd.name.clone();
            if !bounds.is_empty() {
                result.push_str(": ");
                result.push_str(&plus_separated(bounds, pretty_generic_bound));
            }
            if let Some(default) = default {
                result.push_str(" = ");
                result.push_str(&pretty_type(default));
            }
            Some(result)
        }
        GenericParamDefKind::Const { type_, default } => {
            let mut result = format!("const {}: {}", gpd.name, pretty_type(type_));
            if let Some(default) = default {
                result.push_str(" = ");
                result.push_str(default);
            }
            Some(result)
        }
    }
}

fn pretty_generic_bound(bound: &GenericBound) -> String {
    match bound {
        GenericBound::TraitBound { trait_, generic_params, modifier } => {
            let mut result = format!("{}{}", pretty_hrtb(generic_params), pretty_path(trait_));
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
                comma_separated(use_, |arg| match arg {
                    PreciseCapturingArg::Lifetime(lifetime) => lifetime,
                    PreciseCapturingArg::Param(param) => param,
                })
            )
        }
    }
}

fn pretty_where_predicates(predicate: &WherePredicate) -> String {
    match predicate {
        WherePredicate::BoundPredicate { type_, bounds, generic_params } => {
            format!(
                "{}{}: {}",
                pretty_hrtb(generic_params),
                pretty_type(type_),
                plus_separated(bounds, pretty_generic_bound),
            )
        }
        WherePredicate::LifetimePredicate { lifetime, outlives } => {
            format!("{lifetime}: {}", plus_separated(outlives, identity))
        }
        WherePredicate::EqPredicate { lhs, rhs } => {
            format!("{} = {}", pretty_type(lhs), pretty_term(rhs))
        }
    }
}

fn pretty_function_header(header: &FunctionHeader) -> String {
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

    result
}

fn pretty_function_signature(sig: &FunctionSignature) -> String {
    let mut result = format!(
        "({}",
        comma_separated(&sig.inputs, |(name, ty)| format!("{name}: {}", pretty_type(ty)))
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
        result.push_str(&pretty_type(&output));
    }
    result
}

fn comma_separated<T, I, F, R>(iter: I, func: F) -> String
where
    I: IntoIterator<Item = T>,
    R: SeparatedInput,
    F: Fn(T) -> R,
{
    separated(", ", iter, func)
}

fn plus_separated<T, I, F, R>(iter: I, func: F) -> String
where
    I: IntoIterator<Item = T>,
    R: SeparatedInput,
    F: Fn(T) -> R,
{
    separated(" + ", iter, func)
}

fn separated<T, I, F, R>(separator: &str, iter: I, func: F) -> String
where
    I: IntoIterator<Item = T>,
    R: SeparatedInput,
    F: Fn(T) -> R,
{
    let mut result = String::new();
    let mut first = true;
    for item in iter {
        let item_result = func(item);
        let Some(item) = item_result.into_option_str() else { continue };
        if first {
            first = false;
        } else {
            result.push_str(separator);
        }
        result.push_str(item);
    }
    result
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
