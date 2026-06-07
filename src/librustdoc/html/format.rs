//! HTML formatting module
//!
//! This module contains a large number of `Display` implementations for
//! various types in `rustdoc::clean`.
//!
//! These implementations all emit HTML. As an internal implementation detail,
//! some of them support an alternate format that emits plain text.

use std::cmp::Ordering;
use std::fmt::{self, Display, Write};
use std::{iter, slice};

use itertools::{Either, Itertools};
use rustc_abi::ExternAbi;
use rustc_ast::join_path_syms;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, MacroKinds};
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::{ConstStability, StabilityLevel, StableSince};
use rustc_metadata::creader::CStore;
use rustc_middle::ty::{self, TyCtxt, TypingMode};
use rustc_span::symbol::kw;
use rustc_span::{Ident, Symbol};
use tracing::{debug, trace};

use super::url_parts_builder::UrlPartsBuilder;
use crate::clean::types::ExternalLocation;
use crate::clean::utils::find_nearest_parent_module;
use crate::clean::{self, ExternalCrate, PrimitiveType};
use crate::display::{Joined as _, MaybeDisplay as _, WithOpts, Wrapped};
use crate::formats::cache::Cache;
use crate::formats::item_type::ItemType;
use crate::html::escape::{Escape, EscapeBodyText};
use crate::html::render::Context;
use crate::passes::collect_intra_doc_links::UrlFragment;

#[derive(Clone, Copy, Default)]
struct TypePrintOpts<'a> {
    use_absolute: bool,
    disambiguator: Option<&'a ImplPathDisambiguator>,
}

impl<'a> TypePrintOpts<'a> {
    fn nested(self) -> Self {
        Self { use_absolute: false, ..self }
    }

    fn use_absolute_for_path(self, path: &clean::Path) -> bool {
        self.use_absolute
            || self
                .disambiguator
                .is_some_and(|disambiguator| disambiguator.should_disambiguate(path))
    }
}

#[derive(Default)]
struct ImplPathDisambiguator {
    paths: FxHashMap<Symbol, Vec<DefId>>,
    has_ambiguity: bool,
}

impl ImplPathDisambiguator {
    fn new(impl_: &clean::Impl) -> Option<Self> {
        let mut disambiguator = Self::default();

        if let Some(trait_) = &impl_.trait_ {
            disambiguator.add_path(trait_);
        }
        if let Some(ty) = impl_.kind.as_blanket_ty() {
            disambiguator.add_type(ty);
        } else {
            disambiguator.add_type(&impl_.for_);
        }

        disambiguator.has_ambiguity.then_some(disambiguator)
    }

    fn should_disambiguate(&self, path: &clean::Path) -> bool {
        !path.is_assoc_ty()
            && path.last_opt().is_some_and(|last| {
                self.paths
                    .get(&last)
                    .is_some_and(|def_ids| def_ids.len() > 1 && def_ids.contains(&path.def_id()))
            })
    }

    fn add_path(&mut self, path: &clean::Path) {
        if !path.is_assoc_ty()
            && let Some(last) = path.last_opt()
        {
            let did = path.def_id();
            let def_ids = self.paths.entry(last).or_default();
            if !def_ids.contains(&did) {
                def_ids.push(did);
                self.has_ambiguity |= def_ids.len() > 1;
            }
        }

        for segment in &path.segments {
            self.add_generic_args(&segment.args);
        }
    }

    fn add_poly_trait(&mut self, poly_trait: &clean::PolyTrait) {
        self.add_generic_param_defs(&poly_trait.generic_params);
        self.add_path(&poly_trait.trait_);
    }

    fn add_type(&mut self, ty: &clean::Type) {
        match ty {
            clean::Type::Path { path } => self.add_path(path),
            clean::Type::DynTrait(bounds, _) => {
                for bound in bounds {
                    self.add_poly_trait(bound);
                }
            }
            clean::Type::BareFunction(decl) => {
                self.add_generic_param_defs(&decl.generic_params);
                self.add_fn_decl(&decl.decl);
            }
            clean::Type::Tuple(types) => {
                for ty in types {
                    self.add_type(ty);
                }
            }
            clean::Type::Slice(ty)
            | clean::Type::Array(ty, _)
            | clean::Type::Pat(ty, _)
            | clean::Type::FieldOf(ty, _)
            | clean::Type::RawPointer(_, ty)
            | clean::Type::BorrowedRef { type_: ty, .. } => self.add_type(ty),
            clean::Type::QPath(qpath) => {
                self.add_type(&qpath.self_type);
                if let Some(trait_) = &qpath.trait_ {
                    self.add_path(trait_);
                }
                self.add_generic_args(&qpath.assoc.args);
            }
            clean::Type::ImplTrait(bounds) => self.add_generic_bounds(bounds),
            clean::Type::UnsafeBinder(binder) => {
                self.add_generic_param_defs(&binder.generic_params);
                self.add_type(&binder.ty);
            }
            _ => {}
        }
    }

    fn add_generic_args(&mut self, generic_args: &clean::GenericArgs) {
        match generic_args {
            clean::GenericArgs::AngleBracketed { args, constraints } => {
                for arg in args {
                    if let clean::GenericArg::Type(ty) = arg {
                        self.add_type(ty);
                    }
                }
                for constraint in constraints {
                    self.add_assoc_item_constraint(constraint);
                }
            }
            clean::GenericArgs::Parenthesized { inputs, output } => {
                for ty in inputs {
                    self.add_type(ty);
                }
                if let Some(ty) = output {
                    self.add_type(ty);
                }
            }
            _ => {}
        }
    }

    fn add_assoc_item_constraint(&mut self, constraint: &clean::AssocItemConstraint) {
        self.add_generic_args(&constraint.assoc.args);
        match &constraint.kind {
            clean::AssocItemConstraintKind::Equality { term } => self.add_term(term),
            clean::AssocItemConstraintKind::Bound { bounds } => self.add_generic_bounds(bounds),
        }
    }

    fn add_term(&mut self, term: &clean::Term) {
        if let clean::Term::Type(ty) = term {
            self.add_type(ty);
        }
    }

    fn add_generic_bounds(&mut self, bounds: &[clean::GenericBound]) {
        for bound in bounds {
            if let clean::GenericBound::TraitBound(poly_trait, _) = bound {
                self.add_poly_trait(poly_trait);
            }
        }
    }

    fn add_generic_param_defs(&mut self, params: &[clean::GenericParamDef]) {
        for param in params {
            match &param.kind {
                clean::GenericParamDefKind::Lifetime { .. } => {}
                clean::GenericParamDefKind::Type { bounds, default, .. } => {
                    self.add_generic_bounds(bounds);
                    if let Some(ty) = default {
                        self.add_type(ty);
                    }
                }
                clean::GenericParamDefKind::Const { ty, .. } => self.add_type(ty),
            }
        }
    }

    fn add_fn_decl(&mut self, decl: &clean::FnDecl) {
        for input in &decl.inputs {
            self.add_type(&input.type_);
        }
        self.add_type(&decl.output);
    }
}

pub(crate) fn print_generic_bounds(
    bounds: &[clean::GenericBound],
    cx: &Context<'_>,
) -> impl Display {
    print_generic_bounds_with_opts(bounds, cx, TypePrintOpts::default())
}

fn print_generic_bounds_with_opts(
    bounds: &[clean::GenericBound],
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| {
        let mut bounds_dup = FxHashSet::default();

        bounds
            .iter()
            .filter(move |b| bounds_dup.insert(*b))
            .map(|bound| print_generic_bound_with_opts(bound, cx, opts))
            .joined(" + ", f)
    })
}

fn print_generic_param_def_with_opts(
    generic_param: &clean::GenericParamDef,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| match &generic_param.kind {
        clean::GenericParamDefKind::Lifetime { outlives } => {
            write!(f, "{}", generic_param.name)?;

            if !outlives.is_empty() {
                f.write_str(": ")?;
                outlives.iter().map(|lt| print_lifetime(lt)).joined(" + ", f)?;
            }

            Ok(())
        }
        clean::GenericParamDefKind::Type { bounds, default, .. } => {
            f.write_str(generic_param.name.as_str())?;

            if !bounds.is_empty() {
                f.write_str(": ")?;
                print_generic_bounds_with_opts(bounds, cx, opts).fmt(f)?;
            }

            if let Some(ty) = default {
                f.write_str(" = ")?;
                fmt_type(ty, f, opts.nested(), cx)?;
            }

            Ok(())
        }
        clean::GenericParamDefKind::Const { ty, default, .. } => {
            write!(f, "const {}: ", generic_param.name)?;
            fmt_type(ty, f, opts.nested(), cx)?;

            if let Some(default) = default {
                f.write_str(" = ")?;
                if f.alternate() {
                    write!(f, "{default}")?;
                } else {
                    write!(f, "{}", Escape(default))?;
                }
            }

            Ok(())
        }
    })
}

pub(crate) fn print_generics(generics: &clean::Generics, cx: &Context<'_>) -> impl Display {
    print_generics_with_opts(generics, cx, TypePrintOpts::default())
}

fn print_generics_with_opts(
    generics: &clean::Generics,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    let mut real_params = generics.params.iter().filter(|p| !p.is_synthetic_param()).peekable();
    if real_params.peek().is_none() {
        None
    } else {
        Some(Wrapped::with_angle_brackets().wrap_fn(move |f| {
            real_params
                .clone()
                .map(|g| print_generic_param_def_with_opts(g, cx, opts))
                .joined(", ", f)
        }))
    }
    .maybe_display()
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ending {
    Newline,
    NoNewline,
}

fn print_where_predicate_with_opts(
    predicate: &clean::WherePredicate,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| {
        match predicate {
            clean::WherePredicate::BoundPredicate { ty, bounds, bound_params } => {
                print_higher_ranked_params_with_space(bound_params, cx, "for", opts).fmt(f)?;
                fmt_type(ty, f, opts.nested(), cx)?;
                f.write_str(":")?;
                if !bounds.is_empty() {
                    f.write_str(" ")?;
                    print_generic_bounds_with_opts(bounds, cx, opts).fmt(f)?;
                }
                Ok(())
            }
            clean::WherePredicate::RegionPredicate { lifetime, bounds } => {
                // We don't need to check `alternate` since we can be certain that neither
                // the lifetime nor the bounds contain any characters which need escaping.
                write!(f, "{}:", print_lifetime(lifetime))?;
                if !bounds.is_empty() {
                    write!(f, " {}", print_generic_bounds_with_opts(bounds, cx, opts))?;
                }
                Ok(())
            }
            clean::WherePredicate::EqPredicate { lhs, rhs } => {
                let fmt_opts = WithOpts::from(f);
                write!(
                    f,
                    "{} == {}",
                    fmt_opts.display(print_qpath_data_with_opts(lhs, cx, opts)),
                    fmt_opts.display(print_term_with_opts(rhs, cx, opts)),
                )
            }
        }
    })
}

/// * The Generics from which to emit a where-clause.
/// * The number of spaces to indent each line with.
/// * Whether the where-clause needs to add a comma and newline after the last bound.
pub(crate) fn print_where_clause(
    gens: &clean::Generics,
    cx: &Context<'_>,
    indent: usize,
    ending: Ending,
) -> Option<impl Display> {
    print_where_clause_with_opts(gens, cx, indent, ending, TypePrintOpts::default())
}

fn print_where_clause_with_opts(
    gens: &clean::Generics,
    cx: &Context<'_>,
    indent: usize,
    ending: Ending,
    opts: TypePrintOpts<'_>,
) -> Option<impl Display> {
    if gens.where_predicates.is_empty() {
        return None;
    }

    Some(fmt::from_fn(move |f| {
        let where_preds = fmt::from_fn(|f| {
            gens.where_predicates
                .iter()
                .map(|predicate| {
                    fmt::from_fn(|f| {
                        if f.alternate() {
                            f.write_str(" ")?;
                        } else {
                            f.write_str("\n")?;
                        }
                        print_where_predicate_with_opts(predicate, cx, opts).fmt(f)
                    })
                })
                .joined(",", f)
        });

        let clause = if f.alternate() {
            if ending == Ending::Newline {
                format!(" where{where_preds:#},")
            } else {
                format!(" where{where_preds:#}")
            }
        } else {
            let mut br_with_padding = String::with_capacity(6 * indent + 28);
            br_with_padding.push('\n');

            let where_indent = 3;
            let padding_amount = if ending == Ending::Newline {
                indent + 4
            } else if indent == 0 {
                4
            } else {
                indent + where_indent + "where ".len()
            };

            for _ in 0..padding_amount {
                br_with_padding.push(' ');
            }
            let where_preds = where_preds.to_string().replace('\n', &br_with_padding);

            if ending == Ending::Newline {
                let mut clause = " ".repeat(indent.saturating_sub(1));
                write!(clause, "<div class=\"where\">where{where_preds},</div>")?;
                clause
            } else {
                // insert a newline after a single space but before multiple spaces at the start
                if indent == 0 {
                    format!("\n<span class=\"where\">where{where_preds}</span>")
                } else {
                    // put the first one on the same line as the 'where' keyword
                    let where_preds = where_preds.replacen(&br_with_padding, " ", 1);

                    let mut clause = br_with_padding;
                    // +1 is for `\n`.
                    clause.truncate(indent + 1 + where_indent);

                    write!(clause, "<span class=\"where\">where{where_preds}</span>")?;
                    clause
                }
            }
        };
        write!(f, "{clause}")
    }))
}

#[inline]
pub(crate) fn print_lifetime(lt: &clean::Lifetime) -> &str {
    lt.0.as_str()
}

pub(crate) fn print_constant_kind(
    constant_kind: &clean::ConstantKind,
    tcx: TyCtxt<'_>,
) -> impl Display {
    let expr = constant_kind.expr(tcx);
    fmt::from_fn(
        move |f| {
            if f.alternate() { f.write_str(&expr) } else { write!(f, "{}", Escape(&expr)) }
        },
    )
}

fn print_poly_trait_with_opts(
    poly_trait: &clean::PolyTrait,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| {
        print_higher_ranked_params_with_space(&poly_trait.generic_params, cx, "for", opts)
            .fmt(f)?;
        print_path_with_opts(&poly_trait.trait_, cx, opts.nested()).fmt(f)
    })
}

pub(crate) fn print_generic_bound(
    generic_bound: &clean::GenericBound,
    cx: &Context<'_>,
) -> impl Display {
    print_generic_bound_with_opts(generic_bound, cx, TypePrintOpts::default())
}

fn print_generic_bound_with_opts(
    generic_bound: &clean::GenericBound,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| match generic_bound {
        clean::GenericBound::Outlives(lt) => f.write_str(print_lifetime(lt)),
        clean::GenericBound::TraitBound(ty, modifiers) => {
            // `const` and `[const]` trait bounds are experimental; don't render them.
            let hir::TraitBoundModifiers { polarity, constness: _ } = modifiers;
            f.write_str(match polarity {
                hir::BoundPolarity::Positive => "",
                hir::BoundPolarity::Maybe(_) => "?",
                hir::BoundPolarity::Negative(_) => "!",
            })?;
            print_poly_trait_with_opts(ty, cx, opts).fmt(f)
        }
        clean::GenericBound::Use(args) => {
            f.write_str("use")?;
            Wrapped::with_angle_brackets()
                .wrap_fn(|f| args.iter().map(|arg| arg.name()).joined(", ", f))
                .fmt(f)
        }
    })
}

fn print_generic_args_with_opts(
    generic_args: &clean::GenericArgs,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| {
        match generic_args {
            clean::GenericArgs::AngleBracketed { args, constraints } => {
                if !args.is_empty() || !constraints.is_empty() {
                    Wrapped::with_angle_brackets()
                        .wrap_fn(|f| {
                            [Either::Left(args), Either::Right(constraints)]
                                .into_iter()
                                .flat_map(Either::factor_into_iter)
                                .map(|either| {
                                    either.map_either(
                                        |arg| print_generic_arg_with_opts(arg, cx, opts.nested()),
                                        |constraint| {
                                            print_assoc_item_constraint_with_opts(
                                                constraint,
                                                cx,
                                                opts.nested(),
                                            )
                                        },
                                    )
                                })
                                .joined(", ", f)
                        })
                        .fmt(f)?;
                }
            }
            clean::GenericArgs::Parenthesized { inputs, output } => {
                Wrapped::with_parens()
                    .wrap_fn(|f| {
                        inputs
                            .iter()
                            .map(|ty| fmt::from_fn(move |f| fmt_type(ty, f, opts.nested(), cx)))
                            .joined(", ", f)
                    })
                    .fmt(f)?;
                if let Some(ref ty) = *output {
                    f.write_str(if f.alternate() { " -> " } else { " -&gt; " })?;
                    fmt_type(ty, f, opts.nested(), cx)?;
                }
            }
            clean::GenericArgs::ReturnTypeNotation => {
                f.write_str("(..)")?;
            }
        }
        Ok(())
    })
}

// Possible errors when computing href link source for a `DefId`
#[derive(PartialEq, Eq)]
pub(crate) enum HrefError {
    /// This item is known to rustdoc, but from a crate that does not have documentation generated.
    ///
    /// This can only happen for non-local items.
    ///
    /// # Example
    ///
    /// Crate `a` defines a public trait and crate `b` – the target crate that depends on `a` –
    /// implements it for a local type.
    /// We document `b` but **not** `a` (we only _build_ the latter – with `rustc`):
    ///
    /// ```sh
    /// rustc a.rs --crate-type=lib
    /// rustdoc b.rs --crate-type=lib --extern=a=liba.rlib
    /// ```
    ///
    /// Now, the associated items in the trait impl want to link to the corresponding item in the
    /// trait declaration (see `html::render::assoc_href_attr`) but it's not available since their
    /// *documentation (was) not built*.
    DocumentationNotBuilt,
    /// This can only happen for non-local items when `--document-private-items` is not passed.
    Private,
    // Not in external cache, href link should be in same page
    NotInExternalCache,
    /// Refers to an unnamable item, such as one defined within a function or const block.
    UnnamableItem,
}

/// Type representing information of an `href` attribute.
pub(crate) struct HrefInfo {
    /// URL to the item page.
    pub(crate) url: String,
    /// Kind of the item (used to generate the `title` attribute).
    pub(crate) kind: ItemType,
    /// Rust path to the item (used to generate the `title` attribute).
    pub(crate) rust_path: Vec<Symbol>,
}

/// This function is to get the external macro path because they are not in the cache used in
/// `href_with_root_path`.
fn generate_macro_def_id_path(
    def_id: DefId,
    cx: &Context<'_>,
    root_path: Option<&str>,
) -> Result<HrefInfo, HrefError> {
    let tcx = cx.tcx();
    let crate_name = tcx.crate_name(def_id.krate);
    let cache = cx.cache();

    let cstore = CStore::from_tcx(tcx);
    // We need this to prevent a `panic` when this function is used from intra doc links...
    if !cstore.has_crate_data(def_id.krate) {
        debug!("No data for crate {crate_name}");
        return Err(HrefError::NotInExternalCache);
    }
    let DefKind::Macro(kinds) = tcx.def_kind(def_id) else {
        unreachable!();
    };
    let item_type = if kinds == MacroKinds::DERIVE {
        ItemType::ProcDerive
    } else if kinds == MacroKinds::ATTR {
        ItemType::ProcAttribute
    } else {
        ItemType::Macro
    };
    let path = clean::inline::get_item_path(tcx, def_id, item_type);
    // The minimum we can have is the crate name followed by the macro name. If shorter, then
    // it means that `relative` was empty, which is an error.
    let [module_path @ .., last] = path.as_slice() else {
        debug!("macro path is empty!");
        return Err(HrefError::NotInExternalCache);
    };
    if module_path.is_empty() {
        debug!("macro path too short: missing crate prefix (got 1 element, need at least 2)");
        return Err(HrefError::NotInExternalCache);
    }

    let url = match cache.extern_locations[&def_id.krate] {
        ExternalLocation::Remote { ref url, is_absolute } => {
            let mut prefix = remote_url_prefix(url, is_absolute, cx.current.len());
            prefix.extend(module_path.iter().copied());
            prefix.push_fmt(format_args!("{}.{last}.html", item_type.as_str()));
            prefix.finish()
        }
        ExternalLocation::Local => {
            // `root_path` always end with a `/`.
            format!(
                "{root_path}{path}/{item_type}.{last}.html",
                root_path = root_path.unwrap_or(""),
                path = fmt::from_fn(|f| module_path.iter().joined("/", f)),
                item_type = item_type.as_str(),
            )
        }
        ExternalLocation::Unknown => {
            debug!("crate {crate_name} not in cache when linkifying macros");
            return Err(HrefError::NotInExternalCache);
        }
    };
    Ok(HrefInfo { url, kind: item_type, rust_path: path })
}

fn generate_item_def_id_path(
    mut def_id: DefId,
    original_def_id: DefId,
    cx: &Context<'_>,
    root_path: Option<&str>,
) -> Result<HrefInfo, HrefError> {
    use rustc_middle::traits::ObligationCause;
    use rustc_trait_selection::infer::TyCtxtInferExt;
    use rustc_trait_selection::traits::query::normalize::QueryNormalizeExt;

    let tcx = cx.tcx();
    let crate_name = tcx.crate_name(def_id.krate);
    let mut prim = None;

    // No need to try to infer the actual parent item if it's not an associated item from the `impl`
    // block.
    if def_id != original_def_id && matches!(tcx.def_kind(def_id), DefKind::Impl { .. }) {
        let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
        let ty = tcx.type_of(def_id);
        let ty = infcx
            .at(&ObligationCause::dummy(), tcx.param_env(def_id))
            .query_normalize(ty::Binder::dummy(ty.instantiate_identity().skip_norm_wip()))
            .map(|resolved| infcx.resolve_vars_if_possible(resolved.value).skip_binder())
            .unwrap_or(ty.skip_binder());
        if let Some(new_def_id) = ty.ty_adt_def().map(|adt| adt.did()) {
            def_id = new_def_id;
        } else {
            prim = PrimitiveType::from_ty(ty);
        }
    }

    let mut fqp = vec![crate_name];
    let shortty = if let Some(prim) = prim {
        fqp.push(prim.as_sym());
        ItemType::Primitive
    } else {
        fqp.append(&mut clean::inline::item_relative_path(tcx, def_id));
        ItemType::from_def_id(def_id, tcx)
    };
    let module_fqp = to_module_fqp(shortty, &fqp);

    let (parts, is_absolute) = url_parts(cx.cache(), def_id, module_fqp, &cx.current)?;
    let mut url = make_href(root_path, shortty, parts, &fqp, is_absolute);

    if def_id != original_def_id {
        let kind = ItemType::from_def_id(original_def_id, tcx);
        url = format!("{url}#{kind}.{}", tcx.item_name(original_def_id))
    };
    Ok(HrefInfo { url, kind: shortty, rust_path: fqp })
}

/// Checks if the given defid refers to an item that is unnamable, such as one defined in a const block.
fn is_unnamable(tcx: TyCtxt<'_>, did: DefId) -> bool {
    let mut cur_did = did;
    while let Some(parent) = tcx.opt_parent(cur_did) {
        match tcx.def_kind(parent) {
            // items defined in these can be linked to, as long as they are visible
            DefKind::Mod | DefKind::ForeignMod => cur_did = parent,
            // items in impls can be linked to,
            // as long as we can link to the item the impl is on.
            // since associated traits are not a thing,
            // it should not be possible to refer to an impl item if
            // the base type is not namable.
            DefKind::Impl { .. } => return false,
            // everything else does not have docs generated for it
            _ => return true,
        }
    }
    return false;
}

fn to_module_fqp(shortty: ItemType, fqp: &[Symbol]) -> &[Symbol] {
    if shortty == ItemType::Module { fqp } else { &fqp[..fqp.len() - 1] }
}

fn remote_url_prefix(url: &str, is_absolute: bool, depth: usize) -> UrlPartsBuilder {
    let url = url.trim_end_matches('/');
    if is_absolute {
        UrlPartsBuilder::singleton(url)
    } else {
        let extra = depth.saturating_sub(1);
        let mut b: UrlPartsBuilder = iter::repeat_n("..", extra).collect();
        b.push(url);
        b
    }
}

fn url_parts(
    cache: &Cache,
    def_id: DefId,
    module_fqp: &[Symbol],
    relative_to: &[Symbol],
) -> Result<(UrlPartsBuilder, bool), HrefError> {
    match cache.extern_locations[&def_id.krate] {
        ExternalLocation::Remote { ref url, is_absolute } => {
            let mut builder = remote_url_prefix(url, is_absolute, relative_to.len());
            builder.extend(module_fqp.iter().copied());
            Ok((builder, is_absolute))
        }
        ExternalLocation::Local => Ok((href_relative_parts(module_fqp, relative_to), false)),
        ExternalLocation::Unknown => Err(HrefError::DocumentationNotBuilt),
    }
}

fn make_href(
    root_path: Option<&str>,
    shortty: ItemType,
    mut url_parts: UrlPartsBuilder,
    fqp: &[Symbol],
    is_absolute: bool,
) -> String {
    // FIXME: relative extern URLs may break when prefixed with root_path
    if !is_absolute && let Some(root_path) = root_path {
        let root = root_path.trim_end_matches('/');
        url_parts.push_front(root);
    }
    debug!(?url_parts);
    match shortty {
        ItemType::Module => {
            url_parts.push("index.html");
        }
        _ => {
            let last = fqp.last().unwrap();
            url_parts.push_fmt(format_args!("{shortty}.{last}.html"));
        }
    }
    url_parts.finish()
}

pub(crate) fn href_with_root_path(
    original_did: DefId,
    cx: &Context<'_>,
    root_path: Option<&str>,
) -> Result<HrefInfo, HrefError> {
    let tcx = cx.tcx();
    let def_kind = tcx.def_kind(original_did);
    let did = match def_kind {
        DefKind::AssocTy | DefKind::AssocFn | DefKind::AssocConst { .. } | DefKind::Variant => {
            // documented on their parent's page
            tcx.parent(original_did)
        }
        // If this a constructor, we get the parent (either a struct or a variant) and then
        // generate the link for this item.
        DefKind::Ctor(..) => return href_with_root_path(tcx.parent(original_did), cx, root_path),
        DefKind::ExternCrate => {
            // Link to the crate itself, not the `extern crate` item.
            if let Some(local_did) = original_did.as_local() {
                tcx.extern_mod_stmt_cnum(local_did).unwrap_or(LOCAL_CRATE).as_def_id()
            } else {
                original_did
            }
        }
        _ => original_did,
    };
    if is_unnamable(cx.tcx(), did) {
        return Err(HrefError::UnnamableItem);
    }
    let cache = cx.cache();
    let relative_to = &cx.current;

    if !original_did.is_local() {
        // If we are generating an href for the "jump to def" feature, then the only case we want
        // to ignore is if the item is `doc(hidden)` because we can't link to it.
        if root_path.is_some() {
            if tcx.is_doc_hidden(original_did) {
                return Err(HrefError::Private);
            }
        } else if !cache.effective_visibilities.is_directly_public(tcx, did)
            && !cache.document_private
            && !cache.primitive_locations.values().any(|&id| id == did)
        {
            return Err(HrefError::Private);
        }
    }

    let (fqp, shortty, url_parts, is_absolute) = match cache.paths.get(&did) {
        Some(&(ref fqp, shortty)) => (
            fqp,
            shortty,
            {
                let module_fqp = to_module_fqp(shortty, fqp.as_slice());
                debug!(?fqp, ?shortty, ?module_fqp);
                href_relative_parts(module_fqp, relative_to)
            },
            false,
        ),
        None => {
            // Associated items are handled differently with "jump to def". The anchor is generated
            // directly here whereas for intra-doc links, we have some extra computation being
            // performed there.
            let def_id_to_get = if root_path.is_some() { original_did } else { did };
            if let Some(&(ref fqp, shortty)) = cache.external_paths.get(&def_id_to_get) {
                let module_fqp = to_module_fqp(shortty, fqp);
                let (parts, is_absolute) = url_parts(cache, did, module_fqp, relative_to)?;
                (fqp, shortty, parts, is_absolute)
            } else if matches!(def_kind, DefKind::Macro(_)) {
                return generate_macro_def_id_path(did, cx, root_path);
            } else if did.is_local() {
                return Err(HrefError::Private);
            } else {
                return generate_item_def_id_path(did, original_did, cx, root_path);
            }
        }
    };
    Ok(HrefInfo {
        url: make_href(root_path, shortty, url_parts, fqp, is_absolute),
        kind: shortty,
        rust_path: fqp.clone(),
    })
}

pub(crate) fn href(did: DefId, cx: &Context<'_>) -> Result<HrefInfo, HrefError> {
    href_with_root_path(did, cx, None)
}

/// Both paths should only be modules.
/// This is because modules get their own directories; that is, `std::vec` and `std::vec::Vec` will
/// both need `../iter/trait.Iterator.html` to get at the iterator trait.
pub(crate) fn href_relative_parts(fqp: &[Symbol], relative_to_fqp: &[Symbol]) -> UrlPartsBuilder {
    for (i, (f, r)) in fqp.iter().zip(relative_to_fqp.iter()).enumerate() {
        // e.g. linking to std::iter from std::vec (`dissimilar_part_count` will be 1)
        if f != r {
            let dissimilar_part_count = relative_to_fqp.len() - i;
            let fqp_module = &fqp[i..];
            return iter::repeat_n("..", dissimilar_part_count)
                .chain(fqp_module.iter().map(|s| s.as_str()))
                .collect();
        }
    }
    match relative_to_fqp.len().cmp(&fqp.len()) {
        Ordering::Less => {
            // e.g. linking to std::sync::atomic from std::sync
            fqp[relative_to_fqp.len()..fqp.len()].iter().copied().collect()
        }
        Ordering::Greater => {
            // e.g. linking to std::sync from std::sync::atomic
            let dissimilar_part_count = relative_to_fqp.len() - fqp.len();
            iter::repeat_n("..", dissimilar_part_count).collect()
        }
        Ordering::Equal => {
            // linking to the same module
            UrlPartsBuilder::new()
        }
    }
}

pub(crate) fn link_tooltip(
    did: DefId,
    fragment: &Option<UrlFragment>,
    cx: &Context<'_>,
) -> impl fmt::Display {
    fmt::from_fn(move |f| {
        let cache = cx.cache();
        let Some((fqp, shortty)) = cache.paths.get(&did).or_else(|| cache.external_paths.get(&did))
        else {
            return Ok(());
        };
        let fqp = if *shortty == ItemType::Primitive {
            // primitives are documented in a crate, but not actually part of it
            slice::from_ref(fqp.last().unwrap())
        } else {
            fqp
        };
        if let &Some(UrlFragment::Item(id)) = fragment {
            write!(f, "{} ", cx.tcx().def_descr(id))?;
            for component in fqp {
                write!(f, "{component}::")?;
            }
            write!(f, "{}", cx.tcx().item_name(id))?;
        } else if !fqp.is_empty() {
            write!(f, "{shortty} ")?;
            write!(f, "{}", join_path_syms(fqp))?;
        }
        Ok(())
    })
}

/// Used to render a [`clean::Path`].
fn resolved_path(
    w: &mut fmt::Formatter<'_>,
    did: DefId,
    path: &clean::Path,
    print_all: bool,
    opts: TypePrintOpts<'_>,
    cx: &Context<'_>,
) -> fmt::Result {
    let last = path.segments.last().unwrap();

    if print_all {
        for seg in &path.segments[..path.segments.len() - 1] {
            write!(w, "{}::", if seg.name == kw::PathRoot { "" } else { seg.name.as_str() })?;
        }
    }
    if w.alternate() {
        write!(
            w,
            "{}{:#}",
            last.name,
            print_generic_args_with_opts(&last.args, cx, opts.nested())
        )?;
    } else {
        let use_absolute = opts.use_absolute_for_path(path);
        let path = fmt::from_fn(|f| {
            if use_absolute {
                if let Ok(HrefInfo { rust_path, .. }) = href(did, cx) {
                    write!(
                        f,
                        "{path}::{anchor}",
                        path = join_path_syms(&rust_path[..rust_path.len() - 1]),
                        anchor = print_anchor(did, *rust_path.last().unwrap(), cx)
                    )
                } else {
                    write!(f, "{}", last.name)
                }
            } else {
                write!(f, "{}", print_anchor(did, last.name, cx))
            }
        });
        write!(
            w,
            "{path}{args}",
            args = print_generic_args_with_opts(&last.args, cx, opts.nested())
        )?;
    }
    Ok(())
}

fn primitive_link(
    f: &mut fmt::Formatter<'_>,
    prim: clean::PrimitiveType,
    name: fmt::Arguments<'_>,
    cx: &Context<'_>,
) -> fmt::Result {
    primitive_link_fragment(f, prim, name, "", cx)
}

fn primitive_link_fragment(
    f: &mut fmt::Formatter<'_>,
    prim: clean::PrimitiveType,
    name: fmt::Arguments<'_>,
    fragment: &str,
    cx: &Context<'_>,
) -> fmt::Result {
    let m = &cx.cache();
    let mut needs_termination = false;
    if !f.alternate() {
        match m.primitive_locations.get(&prim) {
            Some(&def_id) if def_id.is_local() => {
                let len = cx.current.len();
                let path = fmt::from_fn(|f| {
                    if len == 0 {
                        let cname_sym = ExternalCrate { crate_num: def_id.krate }.name(cx.tcx());
                        write!(f, "{cname_sym}/")?;
                    } else {
                        for _ in 0..(len - 1) {
                            f.write_str("../")?;
                        }
                    }
                    Ok(())
                });
                write!(
                    f,
                    "<a class=\"primitive\" href=\"{path}primitive.{}.html{fragment}\">",
                    prim.as_sym()
                )?;
                needs_termination = true;
            }
            Some(&def_id) => {
                let loc = match m.extern_locations[&def_id.krate] {
                    ExternalLocation::Remote { ref url, is_absolute } => {
                        let cname_sym = ExternalCrate { crate_num: def_id.krate }.name(cx.tcx());
                        let mut builder = remote_url_prefix(url, is_absolute, cx.current.len());
                        builder.push(cname_sym.as_str());
                        Some(builder)
                    }
                    ExternalLocation::Local => {
                        let cname_sym = ExternalCrate { crate_num: def_id.krate }.name(cx.tcx());
                        Some(if cx.current.first() == Some(&cname_sym) {
                            iter::repeat_n("..", cx.current.len() - 1).collect()
                        } else {
                            iter::repeat_n("..", cx.current.len())
                                .chain(iter::once(cname_sym.as_str()))
                                .collect()
                        })
                    }
                    ExternalLocation::Unknown => None,
                };
                if let Some(mut loc) = loc {
                    loc.push_fmt(format_args!("primitive.{}.html", prim.as_sym()));
                    write!(f, "<a class=\"primitive\" href=\"{}{fragment}\">", loc.finish())?;
                    needs_termination = true;
                }
            }
            None => {}
        }
    }
    Display::fmt(&name, f)?;
    if needs_termination {
        write!(f, "</a>")?;
    }
    Ok(())
}

fn print_tybounds(
    bounds: &[clean::PolyTrait],
    lt: &Option<clean::Lifetime>,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| {
        bounds.iter().map(|bound| print_poly_trait_with_opts(bound, cx, opts)).joined(" + ", f)?;
        if let Some(lt) = lt {
            // We don't need to check `alternate` since we can be certain that
            // the lifetime doesn't contain any characters which need escaping.
            write!(f, " + {}", print_lifetime(lt))?;
        }
        Ok(())
    })
}

fn print_higher_ranked_params_with_space(
    params: &[clean::GenericParamDef],
    cx: &Context<'_>,
    keyword: &'static str,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| {
        if !params.is_empty() {
            f.write_str(keyword)?;
            Wrapped::with_angle_brackets()
                .wrap_fn(|f| {
                    params
                        .iter()
                        .map(|lt| print_generic_param_def_with_opts(lt, cx, opts))
                        .joined(", ", f)
                })
                .fmt(f)?;
            f.write_char(' ')?;
        }
        Ok(())
    })
}

pub(crate) fn fragment(did: DefId, tcx: TyCtxt<'_>) -> impl Display {
    fmt::from_fn(move |f| {
        let def_kind = tcx.def_kind(did);
        match def_kind {
            DefKind::AssocTy | DefKind::AssocFn | DefKind::AssocConst { .. } | DefKind::Variant => {
                let item_type = ItemType::from_def_id(did, tcx);
                write!(f, "#{}.{}", item_type.as_str(), tcx.item_name(did))
            }
            DefKind::Field => {
                let parent_def_id = tcx.parent(did);
                f.write_char('#')?;
                if tcx.def_kind(parent_def_id) == DefKind::Variant {
                    write!(f, "variant.{}.field", tcx.item_name(parent_def_id).as_str())?;
                } else {
                    f.write_str("structfield")?;
                };
                write!(f, ".{}", tcx.item_name(did))
            }
            _ => Ok(()),
        }
    })
}

pub(crate) fn print_anchor(did: DefId, text: Symbol, cx: &Context<'_>) -> impl Display {
    fmt::from_fn(move |f| {
        if let Ok(HrefInfo { url, kind, rust_path }) = href(did, cx) {
            write!(
                f,
                r#"<a class="{kind}" href="{url}{anchor}" title="{kind} {path}">{text}</a>"#,
                anchor = fragment(did, cx.tcx()),
                path = join_path_syms(rust_path),
                text = EscapeBodyText(text.as_str()),
            )
        } else {
            f.write_str(text.as_str())
        }
    })
}

fn fmt_type(
    t: &clean::Type,
    f: &mut fmt::Formatter<'_>,
    opts: TypePrintOpts<'_>,
    cx: &Context<'_>,
) -> fmt::Result {
    trace!("fmt_type(t = {t:?})");

    match t {
        clean::Generic(name) => f.write_str(name.as_str()),
        clean::SelfTy => f.write_str("Self"),
        clean::Type::Path { path } => {
            // Paths like `T::Output` and `Self::Output` should be rendered with all segments.
            let did = path.def_id();
            resolved_path(f, did, path, path.is_assoc_ty(), opts, cx)
        }
        clean::DynTrait(bounds, lt) => {
            f.write_str("dyn ")?;
            print_tybounds(bounds, lt, cx, opts.nested()).fmt(f)
        }
        clean::Infer => write!(f, "_"),
        clean::Primitive(clean::PrimitiveType::Never) => {
            primitive_link(f, PrimitiveType::Never, format_args!("!"), cx)
        }
        &clean::Primitive(prim) => primitive_link(f, prim, format_args!("{}", prim.as_sym()), cx),
        clean::BareFunction(decl) => {
            print_higher_ranked_params_with_space(&decl.generic_params, cx, "for", opts).fmt(f)?;
            decl.safety.print_with_space().fmt(f)?;
            print_abi_with_space(decl.abi).fmt(f)?;
            if f.alternate() {
                f.write_str("fn")?;
            } else {
                primitive_link(f, PrimitiveType::Fn, format_args!("fn"), cx)?;
            }
            print_fn_decl_with_opts(&decl.decl, cx, opts.nested()).fmt(f)
        }
        clean::UnsafeBinder(binder) => {
            print_higher_ranked_params_with_space(&binder.generic_params, cx, "unsafe", opts)
                .fmt(f)?;
            fmt_type(&binder.ty, f, opts.nested(), cx)
        }
        clean::Tuple(typs) => match &typs[..] {
            &[] => primitive_link(f, PrimitiveType::Unit, format_args!("()"), cx),
            [one] => {
                if let clean::Generic(name) = one {
                    primitive_link(f, PrimitiveType::Tuple, format_args!("({name},)"), cx)
                } else {
                    write!(f, "(")?;
                    fmt_type(one, f, opts.nested(), cx)?;
                    write!(f, ",)")
                }
            }
            many => {
                let generic_names: Vec<Symbol> = many
                    .iter()
                    .filter_map(|t| match t {
                        clean::Generic(name) => Some(*name),
                        _ => None,
                    })
                    .collect();
                let is_generic = generic_names.len() == many.len();
                if is_generic {
                    primitive_link(
                        f,
                        PrimitiveType::Tuple,
                        format_args!(
                            "{}",
                            Wrapped::with_parens()
                                .wrap_fn(|f| generic_names.iter().joined(", ", f))
                        ),
                        cx,
                    )
                } else {
                    Wrapped::with_parens()
                        .wrap_fn(|f| {
                            many.iter()
                                .map(|item| {
                                    fmt::from_fn(move |f| fmt_type(item, f, opts.nested(), cx))
                                })
                                .joined(", ", f)
                        })
                        .fmt(f)
                }
            }
        },
        clean::Slice(clean::Generic(name)) => {
            primitive_link(f, PrimitiveType::Slice, format_args!("[{name}]"), cx)
        }
        clean::Slice(t) => Wrapped::with_square_brackets()
            .wrap(fmt::from_fn(|f| fmt_type(t, f, opts.nested(), cx)))
            .fmt(f),
        clean::Type::Pat(t, pat) => {
            fmt_type(t, f, opts.nested(), cx)?;
            write!(f, " is {pat}")
        }
        clean::Type::FieldOf(t, field) => {
            write!(f, "field_of!(")?;
            fmt_type(t, f, opts.nested(), cx)?;
            write!(f, ", {field})")
        }
        clean::Array(clean::Generic(name), n) if !f.alternate() => primitive_link(
            f,
            PrimitiveType::Array,
            format_args!("[{name}; {n}]", n = Escape(n)),
            cx,
        ),
        clean::Array(t, n) => Wrapped::with_square_brackets()
            .wrap(fmt::from_fn(|f| {
                fmt_type(t, f, opts.nested(), cx)?;
                f.write_str("; ")?;
                if f.alternate() {
                    f.write_str(n)
                } else {
                    primitive_link(f, PrimitiveType::Array, format_args!("{n}", n = Escape(n)), cx)
                }
            }))
            .fmt(f),
        clean::RawPointer(m, t) => {
            let m = m.ptr_str();

            if matches!(**t, clean::Generic(_)) || t.is_assoc_ty() {
                primitive_link(
                    f,
                    clean::PrimitiveType::RawPointer,
                    format_args!(
                        "*{m} {ty}",
                        ty = WithOpts::from(f).display(fmt::from_fn(|f| fmt_type(
                            t,
                            f,
                            opts.nested(),
                            cx
                        )))
                    ),
                    cx,
                )
            } else {
                primitive_link(f, clean::PrimitiveType::RawPointer, format_args!("*{m} "), cx)?;
                fmt_type(t, f, opts.nested(), cx)
            }
        }
        clean::BorrowedRef { lifetime: l, mutability, type_: ty } => {
            let lt = fmt::from_fn(|f| match l {
                Some(l) => write!(f, "{} ", print_lifetime(l)),
                _ => Ok(()),
            });
            let m = mutability.print_with_space();
            let amp = if f.alternate() { "&" } else { "&amp;" };

            if let clean::Generic(name) = **ty {
                return primitive_link(
                    f,
                    PrimitiveType::Reference,
                    format_args!("{amp}{lt}{m}{name}"),
                    cx,
                );
            }

            write!(f, "{amp}{lt}{m}")?;

            let needs_parens = match **ty {
                clean::DynTrait(ref bounds, ref trait_lt)
                    if bounds.len() > 1 || trait_lt.is_some() =>
                {
                    true
                }
                clean::ImplTrait(ref bounds) if bounds.len() > 1 => true,
                _ => false,
            };
            Wrapped::with_parens().when(needs_parens).wrap_fn(|f| fmt_type(ty, f, opts, cx)).fmt(f)
        }
        clean::ImplTrait(bounds) => {
            f.write_str("impl ")?;
            print_generic_bounds_with_opts(bounds, cx, opts.nested()).fmt(f)
        }
        clean::QPath(qpath) => print_qpath_data_with_opts(qpath, cx, opts.nested()).fmt(f),
    }
}

pub(crate) fn print_type(type_: &clean::Type, cx: &Context<'_>) -> impl Display {
    fmt::from_fn(move |f| fmt_type(type_, f, TypePrintOpts::default(), cx))
}

pub(crate) fn print_path(path: &clean::Path, cx: &Context<'_>) -> impl Display {
    print_path_with_opts(path, cx, TypePrintOpts::default())
}

fn print_path_with_opts(
    path: &clean::Path,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| resolved_path(f, path.def_id(), path, false, opts, cx))
}

fn print_qpath_data_with_opts(
    qpath_data: &clean::QPathData,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    let clean::QPathData { ref assoc, ref self_type, should_fully_qualify, ref trait_ } =
        *qpath_data;

    fmt::from_fn(move |f| {
        // FIXME(inherent_associated_types): Once we support non-ADT self-types (#106719),
        // we need to surround them with angle brackets in some cases (e.g. `<dyn …>::P`).

        if let Some(trait_) = trait_
            && should_fully_qualify
        {
            let fmt_opts = WithOpts::from(f);
            Wrapped::with_angle_brackets()
                .wrap(format_args!(
                    "{} as {}",
                    fmt_opts.display(fmt::from_fn(|f| fmt_type(self_type, f, opts.nested(), cx))),
                    fmt_opts.display(print_path_with_opts(trait_, cx, opts.nested()))
                ))
                .fmt(f)?
        } else {
            fmt_type(self_type, f, opts.nested(), cx)?;
        }
        f.write_str("::")?;
        // It's pretty unsightly to look at `<A as B>::C` in output, and
        // we've got hyperlinking on our side, so try to avoid longer
        // notation as much as possible by making `C` a hyperlink to trait
        // `B` to disambiguate.
        //
        // FIXME: this is still a lossy conversion and there should probably
        //        be a better way of representing this in general? Most of
        //        the ugliness comes from inlining across crates where
        //        everything comes in as a fully resolved QPath (hard to
        //        look at).
        if !f.alternate() {
            // FIXME(inherent_associated_types): We always link to the very first associated
            // type (in respect to source order) that bears the given name (`assoc.name`) and that is
            // affiliated with the computed `DefId`. This is obviously incorrect when we have
            // multiple impl blocks. Ideally, we would thread the `DefId` of the assoc ty itself
            // through here and map it to the corresponding HTML ID that was generated by
            // `render::Context::derive_id` when the impl blocks were rendered.
            // There is no such mapping unfortunately.
            // As a hack, we could badly imitate `derive_id` here by keeping *count* when looking
            // for the assoc ty `DefId` in `tcx.associated_items(self_ty_did).in_definition_order()`
            // considering privacy, `doc(hidden)`, etc.
            // I don't feel like that right now :cold_sweat:.

            let parent_href = match trait_ {
                Some(trait_) => href(trait_.def_id(), cx).ok(),
                None => self_type.def_id(cx.cache()).and_then(|did| href(did, cx).ok()),
            };
            let tcx = cx.tcx();
            let assoc_type_is_hidden = !cx.cache().document_hidden
                && trait_.as_ref().is_some_and(|trait_| {
                    let trait_did = trait_.def_id();
                    tcx.associated_items(trait_did)
                        .find_by_ident_and_kind(
                            tcx,
                            Ident::with_dummy_span(assoc.name),
                            ty::AssocTag::Type,
                            trait_did,
                        )
                        .is_some_and(|assoc_item| tcx.is_doc_hidden(assoc_item.def_id))
                });

            if let Some(HrefInfo { url, rust_path, .. }) = parent_href
                && !assoc_type_is_hidden
            {
                write!(
                    f,
                    "<a class=\"associatedtype\" href=\"{url}#{shortty}.{name}\" \
                                title=\"type {path}::{name}\">{name}</a>",
                    shortty = ItemType::AssocType,
                    name = assoc.name,
                    path = join_path_syms(rust_path),
                )
            } else {
                write!(f, "{}", assoc.name)
            }
        } else {
            write!(f, "{}", assoc.name)
        }?;

        print_generic_args_with_opts(&assoc.args, cx, opts.nested()).fmt(f)
    })
}

pub(crate) fn print_impl(
    impl_: &clean::Impl,
    use_absolute: bool,
    cx: &Context<'_>,
) -> impl Display {
    fmt::from_fn(move |f| {
        fmt_impl(impl_, f, TypePrintOpts { use_absolute, disambiguator: None }, cx)
    })
}

pub(crate) fn print_impl_with_disambiguation(
    impl_: &clean::Impl,
    use_absolute: bool,
    cx: &Context<'_>,
) -> impl Display {
    let disambiguator = ImplPathDisambiguator::new(impl_);

    fmt::from_fn(move |f| {
        fmt_impl(
            impl_,
            f,
            TypePrintOpts { use_absolute, disambiguator: disambiguator.as_ref() },
            cx,
        )
    })
}

fn fmt_impl(
    impl_: &clean::Impl,
    f: &mut fmt::Formatter<'_>,
    opts: TypePrintOpts<'_>,
    cx: &Context<'_>,
) -> fmt::Result {
    f.write_str("impl")?;
    print_generics_with_opts(&impl_.generics, cx, opts.nested()).fmt(f)?;
    f.write_str(" ")?;

    if let Some(ref ty) = impl_.trait_ {
        if impl_.is_negative_trait_impl() {
            f.write_char('!')?;
        }
        if impl_.kind.is_fake_variadic()
            && let Some(generics) = ty.generics()
            && let Ok(inner_type) = generics.exactly_one()
        {
            let last = ty.last();
            if f.alternate() {
                write!(f, "{last}")?;
            } else {
                write!(f, "{}", print_anchor(ty.def_id(), last, cx))?;
            };
            Wrapped::with_angle_brackets()
                .wrap_fn(|f| impl_.print_type(inner_type, f, opts.nested(), cx))
                .fmt(f)?;
        } else {
            print_path_with_opts(ty, cx, opts.nested()).fmt(f)?;
        }
        f.write_str(" for ")?;
    }

    if let Some(ty) = impl_.kind.as_blanket_ty() {
        fmt_type(ty, f, opts, cx)?;
    } else {
        impl_.print_type(&impl_.for_, f, opts, cx)?;
    }

    print_where_clause_with_opts(&impl_.generics, cx, 0, Ending::Newline, opts.nested())
        .maybe_display()
        .fmt(f)
}

impl clean::Impl {
    fn print_type(
        &self,
        type_: &clean::Type,
        f: &mut fmt::Formatter<'_>,
        opts: TypePrintOpts<'_>,
        cx: &Context<'_>,
    ) -> Result<(), fmt::Error> {
        if let clean::Type::Tuple(types) = type_
            && let [clean::Type::Generic(name)] = &types[..]
            && (self.kind.is_fake_variadic() || self.kind.is_auto())
        {
            // Hardcoded anchor library/core/src/primitive_docs.rs
            // Link should match `# Trait implementations`
            primitive_link_fragment(
                f,
                PrimitiveType::Tuple,
                format_args!("({name}₁, {name}₂, …, {name}ₙ)"),
                "#trait-implementations-1",
                cx,
            )?;
        } else if let clean::Type::Array(ty, len) = type_
            && let clean::Type::Generic(name) = &**ty
            && &len[..] == "1"
            && (self.kind.is_fake_variadic() || self.kind.is_auto())
        {
            primitive_link(f, PrimitiveType::Array, format_args!("[{name}; N]"), cx)?;
        } else if let clean::BareFunction(bare_fn) = &type_
            && let [clean::Parameter { type_: clean::Type::Generic(name), .. }] =
                &bare_fn.decl.inputs[..]
            && (self.kind.is_fake_variadic() || self.kind.is_auto())
        {
            // Hardcoded anchor library/core/src/primitive_docs.rs
            // Link should match `# Trait implementations`

            print_higher_ranked_params_with_space(&bare_fn.generic_params, cx, "for", opts)
                .fmt(f)?;
            bare_fn.safety.print_with_space().fmt(f)?;
            print_abi_with_space(bare_fn.abi).fmt(f)?;
            let ellipsis = if bare_fn.decl.c_variadic { ", ..." } else { "" };
            primitive_link_fragment(
                f,
                PrimitiveType::Tuple,
                format_args!("fn({name}₁, {name}₂, …, {name}ₙ{ellipsis})"),
                "#trait-implementations-1",
                cx,
            )?;
            // Write output.
            if !bare_fn.decl.output.is_unit() {
                write!(f, " -> ")?;
                fmt_type(&bare_fn.decl.output, f, opts.nested(), cx)?;
            }
        } else if let clean::Type::Path { path } = type_
            && let Some(generics) = path.generics()
            && let Ok(ty) = generics.exactly_one()
            && self.kind.is_fake_variadic()
        {
            print_anchor(path.def_id(), path.last(), cx).fmt(f)?;
            Wrapped::with_angle_brackets()
                .wrap_fn(|f| self.print_type(ty, f, opts.nested(), cx))
                .fmt(f)?;
        } else {
            fmt_type(type_, f, opts, cx)?;
        }
        Ok(())
    }
}

fn print_params_with_opts(
    params: &[clean::Parameter],
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| {
        params
            .iter()
            .map(|param| {
                fmt::from_fn(|f| {
                    if let Some(name) = param.name {
                        write!(f, "{name}: ")?;
                    }
                    fmt_type(&param.type_, f, opts.nested(), cx)
                })
            })
            .joined(", ", f)
    })
}

// Implements Write but only counts the bytes "written".
struct WriteCounter(usize);

impl std::fmt::Write for WriteCounter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0 += s.len();
        Ok(())
    }
}

// Implements Display by emitting the given number of spaces.
#[derive(Clone, Copy)]
struct Indent(usize);

impl Display for Indent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for _ in 0..self.0 {
            f.write_char(' ')?;
        }
        Ok(())
    }
}

fn print_parameter(parameter: &clean::Parameter, cx: &Context<'_>) -> impl fmt::Display {
    print_parameter_with_opts(parameter, cx, TypePrintOpts::default())
}

fn print_parameter_with_opts(
    parameter: &clean::Parameter,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl fmt::Display {
    fmt::from_fn(move |f| {
        if let Some(self_ty) = parameter.to_receiver() {
            match self_ty {
                clean::SelfTy => f.write_str("self"),
                clean::BorrowedRef { lifetime, mutability, type_: clean::SelfTy } => {
                    f.write_str(if f.alternate() { "&" } else { "&amp;" })?;
                    if let Some(lt) = lifetime {
                        write!(f, "{lt} ", lt = print_lifetime(lt))?;
                    }
                    write!(f, "{mutability}self", mutability = mutability.print_with_space())
                }
                _ => {
                    f.write_str("self: ")?;
                    fmt_type(self_ty, f, opts.nested(), cx)
                }
            }
        } else {
            if parameter.is_const {
                write!(f, "const ")?;
            }
            if let Some(name) = parameter.name {
                write!(f, "{name}: ")?;
            }
            fmt_type(&parameter.type_, f, opts.nested(), cx)
        }
    })
}

fn print_fn_decl_with_opts(
    fn_decl: &clean::FnDecl,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| {
        let ellipsis = if fn_decl.c_variadic { ", ..." } else { "" };
        Wrapped::with_parens()
            .wrap_fn(|f| {
                print_params_with_opts(&fn_decl.inputs, cx, opts).fmt(f)?;
                f.write_str(ellipsis)
            })
            .fmt(f)?;
        fn_decl.print_output_with_opts(cx, opts).fmt(f)
    })
}

/// * `header_len`: The length of the function header and name. In other words, the number of
///   characters in the function declaration up to but not including the parentheses.
///   This is expected to go into a `<pre>`/`code-header` block, so indentation and newlines
///   are preserved.
/// * `indent`: The number of spaces to indent each successive line with, if line-wrapping is
///   necessary.
pub(crate) fn full_print_fn_decl(
    fn_decl: &clean::FnDecl,
    header_len: usize,
    indent: usize,
    cx: &Context<'_>,
) -> impl Display {
    fmt::from_fn(move |f| {
        // First, generate the text form of the declaration, with no line wrapping, and count the bytes.
        let mut counter = WriteCounter(0);
        write!(&mut counter, "{:#}", fmt::from_fn(|f| { fn_decl.inner_full_print(None, f, cx) }))?;
        // If the text form was over 80 characters wide, we will line-wrap our output.
        let line_wrapping_indent = if header_len + counter.0 > 80 { Some(indent) } else { None };
        // Generate the final output. This happens to accept `{:#}` formatting to get textual
        // output but in practice it is only formatted with `{}` to get HTML output.
        fn_decl.inner_full_print(line_wrapping_indent, f, cx)
    })
}

impl clean::FnDecl {
    fn inner_full_print(
        &self,
        // For None, the declaration will not be line-wrapped. For Some(n),
        // the declaration will be line-wrapped, with an indent of n spaces.
        line_wrapping_indent: Option<usize>,
        f: &mut fmt::Formatter<'_>,
        cx: &Context<'_>,
    ) -> fmt::Result {
        Wrapped::with_parens()
            .wrap_fn(|f| {
                if !self.inputs.is_empty() {
                    let line_wrapping_indent = line_wrapping_indent.map(|n| Indent(n + 4));

                    if let Some(indent) = line_wrapping_indent {
                        write!(f, "\n{indent}")?;
                    }

                    let sep = fmt::from_fn(|f| {
                        if let Some(indent) = line_wrapping_indent {
                            write!(f, ",\n{indent}")
                        } else {
                            f.write_str(", ")
                        }
                    });

                    self.inputs.iter().map(|param| print_parameter(param, cx)).joined(sep, f)?;

                    if line_wrapping_indent.is_some() {
                        writeln!(f, ",")?
                    }

                    if self.c_variadic {
                        match line_wrapping_indent {
                            None => write!(f, ", ...")?,
                            Some(indent) => writeln!(f, "{indent}...")?,
                        };
                    }
                }

                if let Some(n) = line_wrapping_indent {
                    write!(f, "{}", Indent(n))?
                }

                Ok(())
            })
            .fmt(f)?;

        self.print_output(cx).fmt(f)
    }

    fn print_output(&self, cx: &Context<'_>) -> impl Display {
        self.print_output_with_opts(cx, TypePrintOpts::default())
    }

    fn print_output_with_opts(&self, cx: &Context<'_>, opts: TypePrintOpts<'_>) -> impl Display {
        fmt::from_fn(move |f| {
            if self.output.is_unit() {
                return Ok(());
            }

            f.write_str(if f.alternate() { " -> " } else { " -&gt; " })?;
            fmt_type(&self.output, f, opts.nested(), cx)
        })
    }
}

pub(crate) fn visibility_print_with_space(item: &clean::Item, cx: &Context<'_>) -> impl Display {
    fmt::from_fn(move |f| {
        let Some(vis) = item.visibility(cx.tcx()) else {
            return Ok(());
        };

        match vis {
            ty::Visibility::Public => f.write_str("pub ")?,
            ty::Visibility::Restricted(vis_did) => {
                // FIXME(camelid): This may not work correctly if `item_did` is a module.
                //                 However, rustdoc currently never displays a module's
                //                 visibility, so it shouldn't matter.
                let parent_module =
                    find_nearest_parent_module(cx.tcx(), item.item_id.expect_def_id());

                if vis_did.is_crate_root() {
                    f.write_str("pub(crate) ")?;
                } else if parent_module == Some(vis_did) {
                    // `pub(in foo)` where `foo` is the parent module
                    // is the same as no visibility modifier; do nothing
                } else if parent_module
                    .and_then(|parent| find_nearest_parent_module(cx.tcx(), parent))
                    == Some(vis_did)
                {
                    f.write_str("pub(super) ")?;
                } else {
                    let path = cx.tcx().def_path(vis_did);
                    debug!("path={path:?}");
                    // modified from `resolved_path()` to work with `DefPathData`
                    let last_name = path.data.last().unwrap().data.get_opt_name().unwrap();
                    let anchor = print_anchor(vis_did, last_name, cx);

                    f.write_str("pub(in ")?;
                    for seg in &path.data[..path.data.len() - 1] {
                        write!(f, "{}::", seg.data.get_opt_name().unwrap())?;
                    }
                    write!(f, "{anchor}) ")?;
                }
            }
        }
        Ok(())
    })
}

pub(crate) trait PrintWithSpace {
    fn print_with_space(&self) -> &str;
}

impl PrintWithSpace for hir::Safety {
    fn print_with_space(&self) -> &str {
        self.prefix_str()
    }
}

impl PrintWithSpace for hir::HeaderSafety {
    fn print_with_space(&self) -> &str {
        match self {
            hir::HeaderSafety::SafeTargetFeatures => "",
            hir::HeaderSafety::Normal(safety) => safety.print_with_space(),
        }
    }
}

impl PrintWithSpace for hir::IsAsync {
    fn print_with_space(&self) -> &str {
        match self {
            hir::IsAsync::Async(_) => "async ",
            hir::IsAsync::NotAsync => "",
        }
    }
}

impl PrintWithSpace for hir::Mutability {
    fn print_with_space(&self) -> &str {
        match self {
            hir::Mutability::Not => "",
            hir::Mutability::Mut => "mut ",
        }
    }
}

pub(crate) fn print_constness_with_space(
    c: &hir::Constness,
    overall_stab: Option<StableSince>,
    const_stab: Option<ConstStability>,
) -> &'static str {
    match c {
        hir::Constness::Const => match (overall_stab, const_stab) {
            // const stable...
            (_, Some(ConstStability { level: StabilityLevel::Stable { .. }, .. }))
            // ...or when feature(staged_api) is not set...
            | (_, None)
            // ...or when const unstable, but overall unstable too
            | (None, Some(ConstStability { level: StabilityLevel::Unstable { .. }, .. })) => {
                "const "
            }
            // const unstable (and overall stable)
            (Some(_), Some(ConstStability { level: StabilityLevel::Unstable { .. }, .. })) => "",
        },
        // not const
        hir::Constness::NotConst => "",
    }
}

pub(crate) fn print_import(import: &clean::Import, cx: &Context<'_>) -> impl Display {
    fmt::from_fn(move |f| match import.kind {
        clean::ImportKind::Simple(name) => {
            if name == import.source.path.last() {
                write!(f, "use {};", print_import_source(&import.source, cx))
            } else {
                write!(
                    f,
                    "use {source} as {name};",
                    source = print_import_source(&import.source, cx)
                )
            }
        }
        clean::ImportKind::Glob => {
            if import.source.path.segments.is_empty() {
                write!(f, "use *;")
            } else {
                write!(f, "use {}::*;", print_import_source(&import.source, cx))
            }
        }
    })
}

fn print_import_source(import_source: &clean::ImportSource, cx: &Context<'_>) -> impl Display {
    fmt::from_fn(move |f| match import_source.did {
        Some(did) => resolved_path(f, did, &import_source.path, true, TypePrintOpts::default(), cx),
        _ => {
            for seg in &import_source.path.segments[..import_source.path.segments.len() - 1] {
                write!(f, "{}::", seg.name)?;
            }
            let name = import_source.path.last();
            if let hir::def::Res::PrimTy(p) = import_source.path.res {
                primitive_link(f, PrimitiveType::from(p), format_args!("{name}"), cx)?;
            } else {
                f.write_str(name.as_str())?;
            }
            Ok(())
        }
    })
}

fn print_assoc_item_constraint_with_opts(
    assoc_item_constraint: &clean::AssocItemConstraint,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| {
        f.write_str(assoc_item_constraint.assoc.name.as_str())?;
        print_generic_args_with_opts(&assoc_item_constraint.assoc.args, cx, opts.nested())
            .fmt(f)?;
        match assoc_item_constraint.kind {
            clean::AssocItemConstraintKind::Equality { ref term } => {
                f.write_str(" = ")?;
                print_term_with_opts(term, cx, opts.nested()).fmt(f)?;
            }
            clean::AssocItemConstraintKind::Bound { ref bounds } => {
                if !bounds.is_empty() {
                    f.write_str(": ")?;
                    print_generic_bounds_with_opts(bounds, cx, opts).fmt(f)?;
                }
            }
        }
        Ok(())
    })
}

pub(crate) fn print_abi_with_space(abi: ExternAbi) -> impl Display {
    fmt::from_fn(move |f| {
        let quot = if f.alternate() { "\"" } else { "&quot;" };
        match abi {
            ExternAbi::Rust => Ok(()),
            abi => write!(f, "extern {0}{1}{0} ", quot, abi.name()),
        }
    })
}

fn print_generic_arg_with_opts(
    generic_arg: &clean::GenericArg,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| match generic_arg {
        clean::GenericArg::Lifetime(lt) => f.write_str(print_lifetime(lt)),
        clean::GenericArg::Type(ty) => fmt_type(ty, f, opts.nested(), cx),
        clean::GenericArg::Const(ct) => print_constant_kind(ct, cx.tcx()).fmt(f),
        clean::GenericArg::Infer => f.write_char('_'),
    })
}

fn print_term_with_opts(
    term: &clean::Term,
    cx: &Context<'_>,
    opts: TypePrintOpts<'_>,
) -> impl Display {
    fmt::from_fn(move |f| match term {
        clean::Term::Type(ty) => fmt_type(ty, f, opts.nested(), cx),
        clean::Term::Constant(ct) => print_constant_kind(ct, cx.tcx()).fmt(f),
    })
}
