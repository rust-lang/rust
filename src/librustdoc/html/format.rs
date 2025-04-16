//! HTML formatting module
//!
//! This module contains a large number of `Display` implementations for
//! various types in `rustdoc::clean`.
//!
//! These implementations all emit HTML. As an internal implementation detail,
//! some of them support an alternate format that emits text, but that should
//! not be used external to this module.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt::{self, Display, Write};
use std::iter::{self, once};
use std::slice;

use itertools::Either;
use rustc_abi::ExternAbi;
use rustc_attr_parsing::{ConstStability, StabilityLevel, StableSince};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_metadata::creader::{CStore, LoadedMacro};
use rustc_middle::ty::{self, TyCtxt, TypingMode};
use rustc_span::symbol::kw;
use rustc_span::{Symbol, sym};
use tracing::{debug, trace};

use super::url_parts_builder::{UrlPartsBuilder, estimate_item_path_byte_length};
use crate::clean::types::ExternalLocation;
use crate::clean::utils::find_nearest_parent_module;
use crate::clean::{self, ExternalCrate, PrimitiveType};
use crate::display::{Joined as _, MaybeDisplay as _};
use crate::formats::cache::Cache;
use crate::formats::item_type::ItemType;
use crate::html::escape::{Escape, EscapeBodyText};
use crate::html::render::Context;
use crate::passes::collect_intra_doc_links::UrlFragment;

pub(crate) fn write_str(s: &mut String, f: fmt::Arguments<'_>) {
    s.write_fmt(f).unwrap();
}

pub(crate) fn print_generic_bounds(
    bounds: &[clean::GenericBound],
    cx: &Context<'_>,
) -> impl Display {
    fmt::from_fn(move |f| {
        let mut bounds_dup = FxHashSet::default();

        bounds
            .iter()
            .filter(move |b| bounds_dup.insert(*b))
            .map(|bound| bound.print(cx))
            .joined(" + ", f)
    })
}

impl clean::GenericParamDef {
    pub(crate) fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| match &self.kind {
            clean::GenericParamDefKind::Lifetime { outlives } => {
                write!(f, "{}", self.name)?;

                if !outlives.is_empty() {
                    f.write_str(": ")?;
                    outlives.iter().map(|lt| lt.print()).joined(" + ", f)?;
                }

                Ok(())
            }
            clean::GenericParamDefKind::Type { bounds, default, .. } => {
                f.write_str(self.name.as_str())?;

                if !bounds.is_empty() {
                    f.write_str(": ")?;
                    print_generic_bounds(bounds, cx).fmt(f)?;
                }

                if let Some(ty) = default {
                    f.write_str(" = ")?;
                    ty.print(cx).fmt(f)?;
                }

                Ok(())
            }
            clean::GenericParamDefKind::Const { ty, default, .. } => {
                write!(f, "const {}: ", self.name)?;
                ty.print(cx).fmt(f)?;

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
}

impl clean::Generics {
    pub(crate) fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| {
            let mut real_params = self.params.iter().filter(|p| !p.is_synthetic_param()).peekable();
            if real_params.peek().is_none() {
                return Ok(());
            }

            let real_params =
                fmt::from_fn(|f| real_params.clone().map(|g| g.print(cx)).joined(", ", f));
            if f.alternate() {
                write!(f, "<{:#}>", real_params)
            } else {
                write!(f, "&lt;{}&gt;", real_params)
            }
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ending {
    Newline,
    NoNewline,
}

fn print_where_predicate(predicate: &clean::WherePredicate, cx: &Context<'_>) -> impl Display {
    fmt::from_fn(move |f| {
        match predicate {
            clean::WherePredicate::BoundPredicate { ty, bounds, bound_params } => {
                print_higher_ranked_params_with_space(bound_params, cx, "for").fmt(f)?;
                ty.print(cx).fmt(f)?;
                f.write_str(":")?;
                if !bounds.is_empty() {
                    f.write_str(" ")?;
                    print_generic_bounds(bounds, cx).fmt(f)?;
                }
                Ok(())
            }
            clean::WherePredicate::RegionPredicate { lifetime, bounds } => {
                // We don't need to check `alternate` since we can be certain that neither
                // the lifetime nor the bounds contain any characters which need escaping.
                write!(f, "{}:", lifetime.print())?;
                if !bounds.is_empty() {
                    write!(f, " {}", print_generic_bounds(bounds, cx))?;
                }
                Ok(())
            }
            clean::WherePredicate::EqPredicate { lhs, rhs } => {
                if f.alternate() {
                    write!(f, "{:#} == {:#}", lhs.print(cx), rhs.print(cx))
                } else {
                    write!(f, "{} == {}", lhs.print(cx), rhs.print(cx))
                }
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
                        print_where_predicate(predicate, cx).fmt(f)
                    })
                })
                .joined(",", f)
        });

        let clause = if f.alternate() {
            if ending == Ending::Newline {
                format!(" where{where_preds},")
            } else {
                format!(" where{where_preds}")
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

impl clean::Lifetime {
    pub(crate) fn print(&self) -> impl Display {
        self.0.as_str()
    }
}

impl clean::ConstantKind {
    pub(crate) fn print(&self, tcx: TyCtxt<'_>) -> impl Display {
        let expr = self.expr(tcx);
        fmt::from_fn(move |f| {
            if f.alternate() { f.write_str(&expr) } else { write!(f, "{}", Escape(&expr)) }
        })
    }
}

impl clean::PolyTrait {
    fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| {
            print_higher_ranked_params_with_space(&self.generic_params, cx, "for").fmt(f)?;
            self.trait_.print(cx).fmt(f)
        })
    }
}

impl clean::GenericBound {
    pub(crate) fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| match self {
            clean::GenericBound::Outlives(lt) => write!(f, "{}", lt.print()),
            clean::GenericBound::TraitBound(ty, modifiers) => {
                // `const` and `~const` trait bounds are experimental; don't render them.
                let hir::TraitBoundModifiers { polarity, constness: _ } = modifiers;
                f.write_str(match polarity {
                    hir::BoundPolarity::Positive => "",
                    hir::BoundPolarity::Maybe(_) => "?",
                    hir::BoundPolarity::Negative(_) => "!",
                })?;
                ty.print(cx).fmt(f)
            }
            clean::GenericBound::Use(args) => {
                if f.alternate() {
                    f.write_str("use<")?;
                } else {
                    f.write_str("use&lt;")?;
                }
                args.iter().map(|arg| arg.name()).joined(", ", f)?;
                if f.alternate() { f.write_str(">") } else { f.write_str("&gt;") }
            }
        })
    }
}

impl clean::GenericArgs {
    fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| {
            match self {
                clean::GenericArgs::AngleBracketed { args, constraints } => {
                    if !args.is_empty() || !constraints.is_empty() {
                        if f.alternate() {
                            f.write_str("<")?;
                        } else {
                            f.write_str("&lt;")?;
                        }

                        [Either::Left(args), Either::Right(constraints)]
                            .into_iter()
                            .flat_map(Either::factor_into_iter)
                            .map(|either| {
                                either.map_either(
                                    |arg| arg.print(cx),
                                    |constraint| constraint.print(cx),
                                )
                            })
                            .joined(", ", f)?;

                        if f.alternate() {
                            f.write_str(">")?;
                        } else {
                            f.write_str("&gt;")?;
                        }
                    }
                }
                clean::GenericArgs::Parenthesized { inputs, output } => {
                    f.write_str("(")?;
                    inputs.iter().map(|ty| ty.print(cx)).joined(", ", f)?;
                    f.write_str(")")?;
                    if let Some(ref ty) = *output {
                        if f.alternate() {
                            write!(f, " -> {:#}", ty.print(cx))?;
                        } else {
                            write!(f, " -&gt; {}", ty.print(cx))?;
                        }
                    }
                }
                clean::GenericArgs::ReturnTypeNotation => {
                    f.write_str("(..)")?;
                }
            }
            Ok(())
        })
    }
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
}

// Panics if `syms` is empty.
pub(crate) fn join_with_double_colon(syms: &[Symbol]) -> String {
    let mut s = String::with_capacity(estimate_item_path_byte_length(syms.len()));
    // NOTE: using `Joined::joined` here causes a noticeable perf regression
    s.push_str(syms[0].as_str());
    for sym in &syms[1..] {
        s.push_str("::");
        s.push_str(sym.as_str());
    }
    s
}

/// This function is to get the external macro path because they are not in the cache used in
/// `href_with_root_path`.
fn generate_macro_def_id_path(
    def_id: DefId,
    cx: &Context<'_>,
    root_path: Option<&str>,
) -> Result<(String, ItemType, Vec<Symbol>), HrefError> {
    let tcx = cx.tcx();
    let crate_name = tcx.crate_name(def_id.krate);
    let cache = cx.cache();

    let fqp = clean::inline::item_relative_path(tcx, def_id);
    let mut relative = fqp.iter().copied();
    let cstore = CStore::from_tcx(tcx);
    // We need this to prevent a `panic` when this function is used from intra doc links...
    if !cstore.has_crate_data(def_id.krate) {
        debug!("No data for crate {crate_name}");
        return Err(HrefError::NotInExternalCache);
    }
    // Check to see if it is a macro 2.0 or built-in macro.
    // More information in <https://rust-lang.github.io/rfcs/1584-macros.html>.
    let is_macro_2 = match cstore.load_macro_untracked(def_id, tcx) {
        // If `def.macro_rules` is `true`, then it's not a macro 2.0.
        LoadedMacro::MacroDef { def, .. } => !def.macro_rules,
        _ => false,
    };

    let mut path = if is_macro_2 {
        once(crate_name).chain(relative).collect()
    } else {
        vec![crate_name, relative.next_back().unwrap()]
    };
    if path.len() < 2 {
        // The minimum we can have is the crate name followed by the macro name. If shorter, then
        // it means that `relative` was empty, which is an error.
        debug!("macro path cannot be empty!");
        return Err(HrefError::NotInExternalCache);
    }

    if let Some(last) = path.last_mut() {
        *last = Symbol::intern(&format!("macro.{last}.html"));
    }

    let url = match cache.extern_locations[&def_id.krate] {
        ExternalLocation::Remote(ref s) => {
            // `ExternalLocation::Remote` always end with a `/`.
            format!("{s}{path}", path = fmt::from_fn(|f| path.iter().joined("/", f)))
        }
        ExternalLocation::Local => {
            // `root_path` always end with a `/`.
            format!(
                "{root_path}{path}",
                root_path = root_path.unwrap_or(""),
                path = fmt::from_fn(|f| path.iter().joined("/", f))
            )
        }
        ExternalLocation::Unknown => {
            debug!("crate {crate_name} not in cache when linkifying macros");
            return Err(HrefError::NotInExternalCache);
        }
    };
    Ok((url, ItemType::Macro, fqp))
}

fn generate_item_def_id_path(
    mut def_id: DefId,
    original_def_id: DefId,
    cx: &Context<'_>,
    root_path: Option<&str>,
    original_def_kind: DefKind,
) -> Result<(String, ItemType, Vec<Symbol>), HrefError> {
    use rustc_middle::traits::ObligationCause;
    use rustc_trait_selection::infer::TyCtxtInferExt;
    use rustc_trait_selection::traits::query::normalize::QueryNormalizeExt;

    let tcx = cx.tcx();
    let crate_name = tcx.crate_name(def_id.krate);

    // No need to try to infer the actual parent item if it's not an associated item from the `impl`
    // block.
    if def_id != original_def_id && matches!(tcx.def_kind(def_id), DefKind::Impl { .. }) {
        let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
        def_id = infcx
            .at(&ObligationCause::dummy(), tcx.param_env(def_id))
            .query_normalize(ty::Binder::dummy(tcx.type_of(def_id).instantiate_identity()))
            .map(|resolved| infcx.resolve_vars_if_possible(resolved.value))
            .ok()
            .and_then(|normalized| normalized.skip_binder().ty_adt_def())
            .map(|adt| adt.did())
            .unwrap_or(def_id);
    }

    let relative = clean::inline::item_relative_path(tcx, def_id);
    let fqp: Vec<Symbol> = once(crate_name).chain(relative).collect();

    let def_kind = tcx.def_kind(def_id);
    let shortty = def_kind.into();
    let module_fqp = to_module_fqp(shortty, &fqp);
    let mut is_remote = false;

    let url_parts = url_parts(cx.cache(), def_id, module_fqp, &cx.current, &mut is_remote)?;
    let (url_parts, shortty, fqp) = make_href(root_path, shortty, url_parts, &fqp, is_remote)?;
    if def_id == original_def_id {
        return Ok((url_parts, shortty, fqp));
    }
    let kind = ItemType::from_def_kind(original_def_kind, Some(def_kind));
    Ok((format!("{url_parts}#{kind}.{}", tcx.item_name(original_def_id)), shortty, fqp))
}

fn to_module_fqp(shortty: ItemType, fqp: &[Symbol]) -> &[Symbol] {
    if shortty == ItemType::Module { fqp } else { &fqp[..fqp.len() - 1] }
}

fn url_parts(
    cache: &Cache,
    def_id: DefId,
    module_fqp: &[Symbol],
    relative_to: &[Symbol],
    is_remote: &mut bool,
) -> Result<UrlPartsBuilder, HrefError> {
    match cache.extern_locations[&def_id.krate] {
        ExternalLocation::Remote(ref s) => {
            *is_remote = true;
            let s = s.trim_end_matches('/');
            let mut builder = UrlPartsBuilder::singleton(s);
            builder.extend(module_fqp.iter().copied());
            Ok(builder)
        }
        ExternalLocation::Local => Ok(href_relative_parts(module_fqp, relative_to).collect()),
        ExternalLocation::Unknown => Err(HrefError::DocumentationNotBuilt),
    }
}

fn make_href(
    root_path: Option<&str>,
    shortty: ItemType,
    mut url_parts: UrlPartsBuilder,
    fqp: &[Symbol],
    is_remote: bool,
) -> Result<(String, ItemType, Vec<Symbol>), HrefError> {
    if !is_remote && let Some(root_path) = root_path {
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
    Ok((url_parts.finish(), shortty, fqp.to_vec()))
}

pub(crate) fn href_with_root_path(
    original_did: DefId,
    cx: &Context<'_>,
    root_path: Option<&str>,
) -> Result<(String, ItemType, Vec<Symbol>), HrefError> {
    let tcx = cx.tcx();
    let def_kind = tcx.def_kind(original_did);
    let did = match def_kind {
        DefKind::AssocTy | DefKind::AssocFn | DefKind::AssocConst | DefKind::Variant => {
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

    let mut is_remote = false;
    let (fqp, shortty, url_parts) = match cache.paths.get(&did) {
        Some(&(ref fqp, shortty)) => (fqp, shortty, {
            let module_fqp = to_module_fqp(shortty, fqp.as_slice());
            debug!(?fqp, ?shortty, ?module_fqp);
            href_relative_parts(module_fqp, relative_to).collect()
        }),
        None => {
            // Associated items are handled differently with "jump to def". The anchor is generated
            // directly here whereas for intra-doc links, we have some extra computation being
            // performed there.
            let def_id_to_get = if root_path.is_some() { original_did } else { did };
            if let Some(&(ref fqp, shortty)) = cache.external_paths.get(&def_id_to_get) {
                let module_fqp = to_module_fqp(shortty, fqp);
                (fqp, shortty, url_parts(cache, did, module_fqp, relative_to, &mut is_remote)?)
            } else if matches!(def_kind, DefKind::Macro(_)) {
                return generate_macro_def_id_path(did, cx, root_path);
            } else if did.is_local() {
                return Err(HrefError::Private);
            } else {
                return generate_item_def_id_path(did, original_did, cx, root_path, def_kind);
            }
        }
    };
    make_href(root_path, shortty, url_parts, fqp, is_remote)
}

pub(crate) fn href(
    did: DefId,
    cx: &Context<'_>,
) -> Result<(String, ItemType, Vec<Symbol>), HrefError> {
    href_with_root_path(did, cx, None)
}

/// Both paths should only be modules.
/// This is because modules get their own directories; that is, `std::vec` and `std::vec::Vec` will
/// both need `../iter/trait.Iterator.html` to get at the iterator trait.
pub(crate) fn href_relative_parts<'fqp>(
    fqp: &'fqp [Symbol],
    relative_to_fqp: &[Symbol],
) -> Box<dyn Iterator<Item = Symbol> + 'fqp> {
    for (i, (f, r)) in fqp.iter().zip(relative_to_fqp.iter()).enumerate() {
        // e.g. linking to std::iter from std::vec (`dissimilar_part_count` will be 1)
        if f != r {
            let dissimilar_part_count = relative_to_fqp.len() - i;
            let fqp_module = &fqp[i..];
            return Box::new(
                iter::repeat_n(sym::dotdot, dissimilar_part_count)
                    .chain(fqp_module.iter().copied()),
            );
        }
    }
    match relative_to_fqp.len().cmp(&fqp.len()) {
        Ordering::Less => {
            // e.g. linking to std::sync::atomic from std::sync
            Box::new(fqp[relative_to_fqp.len()..fqp.len()].iter().copied())
        }
        Ordering::Greater => {
            // e.g. linking to std::sync from std::sync::atomic
            let dissimilar_part_count = relative_to_fqp.len() - fqp.len();
            Box::new(iter::repeat_n(sym::dotdot, dissimilar_part_count))
        }
        Ordering::Equal => {
            // linking to the same module
            Box::new(iter::empty())
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
            fqp.iter().joined("::", f)?;
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
    use_absolute: bool,
    cx: &Context<'_>,
) -> fmt::Result {
    let last = path.segments.last().unwrap();

    if print_all {
        for seg in &path.segments[..path.segments.len() - 1] {
            write!(w, "{}::", if seg.name == kw::PathRoot { "" } else { seg.name.as_str() })?;
        }
    }
    if w.alternate() {
        write!(w, "{}{:#}", last.name, last.args.print(cx))?;
    } else {
        let path = fmt::from_fn(|f| {
            if use_absolute {
                if let Ok((_, _, fqp)) = href(did, cx) {
                    write!(
                        f,
                        "{path}::{anchor}",
                        path = join_with_double_colon(&fqp[..fqp.len() - 1]),
                        anchor = anchor(did, *fqp.last().unwrap(), cx)
                    )
                } else {
                    write!(f, "{}", last.name)
                }
            } else {
                write!(f, "{}", anchor(did, last.name, cx))
            }
        });
        write!(w, "{path}{args}", args = last.args.print(cx))?;
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
                    ExternalLocation::Remote(ref s) => {
                        let cname_sym = ExternalCrate { crate_num: def_id.krate }.name(cx.tcx());
                        let builder: UrlPartsBuilder =
                            [s.as_str().trim_end_matches('/'), cname_sym.as_str()]
                                .into_iter()
                                .collect();
                        Some(builder)
                    }
                    ExternalLocation::Local => {
                        let cname_sym = ExternalCrate { crate_num: def_id.krate }.name(cx.tcx());
                        Some(if cx.current.first() == Some(&cname_sym) {
                            iter::repeat_n(sym::dotdot, cx.current.len() - 1).collect()
                        } else {
                            iter::repeat_n(sym::dotdot, cx.current.len())
                                .chain(iter::once(cname_sym))
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

fn tybounds(
    bounds: &[clean::PolyTrait],
    lt: &Option<clean::Lifetime>,
    cx: &Context<'_>,
) -> impl Display {
    fmt::from_fn(move |f| {
        bounds.iter().map(|bound| bound.print(cx)).joined(" + ", f)?;
        if let Some(lt) = lt {
            // We don't need to check `alternate` since we can be certain that
            // the lifetime doesn't contain any characters which need escaping.
            write!(f, " + {}", lt.print())?;
        }
        Ok(())
    })
}

fn print_higher_ranked_params_with_space(
    params: &[clean::GenericParamDef],
    cx: &Context<'_>,
    keyword: &'static str,
) -> impl Display {
    fmt::from_fn(move |f| {
        if !params.is_empty() {
            f.write_str(keyword)?;
            f.write_str(if f.alternate() { "<" } else { "&lt;" })?;
            params.iter().map(|lt| lt.print(cx)).joined(", ", f)?;
            f.write_str(if f.alternate() { "> " } else { "&gt; " })?;
        }
        Ok(())
    })
}

pub(crate) fn anchor(did: DefId, text: Symbol, cx: &Context<'_>) -> impl Display {
    fmt::from_fn(move |f| {
        let parts = href(did, cx);
        if let Ok((url, short_ty, fqp)) = parts {
            write!(
                f,
                r#"<a class="{short_ty}" href="{url}" title="{short_ty} {path}">{text}</a>"#,
                path = join_with_double_colon(&fqp),
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
    use_absolute: bool,
    cx: &Context<'_>,
) -> fmt::Result {
    trace!("fmt_type(t = {t:?})");

    match *t {
        clean::Generic(name) => f.write_str(name.as_str()),
        clean::SelfTy => f.write_str("Self"),
        clean::Type::Path { ref path } => {
            // Paths like `T::Output` and `Self::Output` should be rendered with all segments.
            let did = path.def_id();
            resolved_path(f, did, path, path.is_assoc_ty(), use_absolute, cx)
        }
        clean::DynTrait(ref bounds, ref lt) => {
            f.write_str("dyn ")?;
            tybounds(bounds, lt, cx).fmt(f)
        }
        clean::Infer => write!(f, "_"),
        clean::Primitive(clean::PrimitiveType::Never) => {
            primitive_link(f, PrimitiveType::Never, format_args!("!"), cx)
        }
        clean::Primitive(prim) => primitive_link(f, prim, format_args!("{}", prim.as_sym()), cx),
        clean::BareFunction(ref decl) => {
            print_higher_ranked_params_with_space(&decl.generic_params, cx, "for").fmt(f)?;
            decl.safety.print_with_space().fmt(f)?;
            print_abi_with_space(decl.abi).fmt(f)?;
            if f.alternate() {
                f.write_str("fn")?;
            } else {
                primitive_link(f, PrimitiveType::Fn, format_args!("fn"), cx)?;
            }
            decl.decl.print(cx).fmt(f)
        }
        clean::UnsafeBinder(ref binder) => {
            print_higher_ranked_params_with_space(&binder.generic_params, cx, "unsafe").fmt(f)?;
            binder.ty.print(cx).fmt(f)
        }
        clean::Tuple(ref typs) => match &typs[..] {
            &[] => primitive_link(f, PrimitiveType::Unit, format_args!("()"), cx),
            [one] => {
                if let clean::Generic(name) = one {
                    primitive_link(f, PrimitiveType::Tuple, format_args!("({name},)"), cx)
                } else {
                    write!(f, "(")?;
                    one.print(cx).fmt(f)?;
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
                            "({})",
                            fmt::from_fn(|f| generic_names.iter().joined(", ", f))
                        ),
                        cx,
                    )
                } else {
                    f.write_str("(")?;
                    many.iter().map(|item| item.print(cx)).joined(", ", f)?;
                    f.write_str(")")
                }
            }
        },
        clean::Slice(ref t) => match **t {
            clean::Generic(name) => {
                primitive_link(f, PrimitiveType::Slice, format_args!("[{name}]"), cx)
            }
            _ => {
                write!(f, "[")?;
                t.print(cx).fmt(f)?;
                write!(f, "]")
            }
        },
        clean::Type::Pat(ref t, ref pat) => {
            fmt::Display::fmt(&t.print(cx), f)?;
            write!(f, " is {pat}")
        }
        clean::Array(ref t, ref n) => match **t {
            clean::Generic(name) if !f.alternate() => primitive_link(
                f,
                PrimitiveType::Array,
                format_args!("[{name}; {n}]", n = Escape(n)),
                cx,
            ),
            _ => {
                write!(f, "[")?;
                t.print(cx).fmt(f)?;
                if f.alternate() {
                    write!(f, "; {n}")?;
                } else {
                    write!(f, "; ")?;
                    primitive_link(
                        f,
                        PrimitiveType::Array,
                        format_args!("{n}", n = Escape(n)),
                        cx,
                    )?;
                }
                write!(f, "]")
            }
        },
        clean::RawPointer(m, ref t) => {
            let m = match m {
                hir::Mutability::Mut => "mut",
                hir::Mutability::Not => "const",
            };

            if matches!(**t, clean::Generic(_)) || t.is_assoc_ty() {
                let ty = t.print(cx);
                if f.alternate() {
                    primitive_link(
                        f,
                        clean::PrimitiveType::RawPointer,
                        format_args!("*{m} {ty:#}"),
                        cx,
                    )
                } else {
                    primitive_link(
                        f,
                        clean::PrimitiveType::RawPointer,
                        format_args!("*{m} {ty}"),
                        cx,
                    )
                }
            } else {
                primitive_link(f, clean::PrimitiveType::RawPointer, format_args!("*{m} "), cx)?;
                t.print(cx).fmt(f)
            }
        }
        clean::BorrowedRef { lifetime: ref l, mutability, type_: ref ty } => {
            let lt = fmt::from_fn(|f| match l {
                Some(l) => write!(f, "{} ", l.print()),
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
            if needs_parens {
                f.write_str("(")?;
            }
            fmt_type(ty, f, use_absolute, cx)?;
            if needs_parens {
                f.write_str(")")?;
            }
            Ok(())
        }
        clean::ImplTrait(ref bounds) => {
            f.write_str("impl ")?;
            print_generic_bounds(bounds, cx).fmt(f)
        }
        clean::QPath(box clean::QPathData {
            ref assoc,
            ref self_type,
            ref trait_,
            should_show_cast,
        }) => {
            // FIXME(inherent_associated_types): Once we support non-ADT self-types (#106719),
            // we need to surround them with angle brackets in some cases (e.g. `<dyn …>::P`).

            if f.alternate() {
                if let Some(trait_) = trait_
                    && should_show_cast
                {
                    write!(f, "<{:#} as {:#}>::", self_type.print(cx), trait_.print(cx))?
                } else {
                    write!(f, "{:#}::", self_type.print(cx))?
                }
            } else {
                if let Some(trait_) = trait_
                    && should_show_cast
                {
                    write!(f, "&lt;{} as {}&gt;::", self_type.print(cx), trait_.print(cx))?
                } else {
                    write!(f, "{}::", self_type.print(cx))?
                }
            };
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

                if let Some((url, _, path)) = parent_href {
                    write!(
                        f,
                        "<a class=\"associatedtype\" href=\"{url}#{shortty}.{name}\" \
                                    title=\"type {path}::{name}\">{name}</a>",
                        shortty = ItemType::AssocType,
                        name = assoc.name,
                        path = join_with_double_colon(&path),
                    )
                } else {
                    write!(f, "{}", assoc.name)
                }
            } else {
                write!(f, "{}", assoc.name)
            }?;

            assoc.args.print(cx).fmt(f)
        }
    }
}

impl clean::Type {
    pub(crate) fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| fmt_type(self, f, false, cx))
    }
}

impl clean::Path {
    pub(crate) fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| resolved_path(f, self.def_id(), self, false, false, cx))
    }
}

impl clean::Impl {
    pub(crate) fn print(&self, use_absolute: bool, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| {
            f.write_str("impl")?;
            self.generics.print(cx).fmt(f)?;
            f.write_str(" ")?;

            if let Some(ref ty) = self.trait_ {
                if self.is_negative_trait_impl() {
                    write!(f, "!")?;
                }
                if self.kind.is_fake_variadic()
                    && let generics = ty.generics()
                    && let &[inner_type] = generics.as_ref().map_or(&[][..], |v| &v[..])
                {
                    let last = ty.last();
                    if f.alternate() {
                        write!(f, "{}<", last)?;
                        self.print_type(inner_type, f, use_absolute, cx)?;
                        write!(f, ">")?;
                    } else {
                        write!(f, "{}&lt;", anchor(ty.def_id(), last, cx))?;
                        self.print_type(inner_type, f, use_absolute, cx)?;
                        write!(f, "&gt;")?;
                    }
                } else {
                    ty.print(cx).fmt(f)?;
                }
                write!(f, " for ")?;
            }

            if let Some(ty) = self.kind.as_blanket_ty() {
                fmt_type(ty, f, use_absolute, cx)?;
            } else {
                self.print_type(&self.for_, f, use_absolute, cx)?;
            }

            print_where_clause(&self.generics, cx, 0, Ending::Newline).maybe_display().fmt(f)
        })
    }
    fn print_type(
        &self,
        type_: &clean::Type,
        f: &mut fmt::Formatter<'_>,
        use_absolute: bool,
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
            && let [clean::Argument { type_: clean::Type::Generic(name), .. }] =
                &bare_fn.decl.inputs.values[..]
            && (self.kind.is_fake_variadic() || self.kind.is_auto())
        {
            // Hardcoded anchor library/core/src/primitive_docs.rs
            // Link should match `# Trait implementations`

            print_higher_ranked_params_with_space(&bare_fn.generic_params, cx, "for").fmt(f)?;
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
                fmt_type(&bare_fn.decl.output, f, use_absolute, cx)?;
            }
        } else if let clean::Type::Path { path } = type_
            && let Some(generics) = path.generics()
            && generics.len() == 1
            && self.kind.is_fake_variadic()
        {
            let ty = generics[0];
            let wrapper = anchor(path.def_id(), path.last(), cx);
            if f.alternate() {
                write!(f, "{wrapper:#}&lt;")?;
            } else {
                write!(f, "{wrapper}<")?;
            }
            self.print_type(ty, f, use_absolute, cx)?;
            if f.alternate() {
                write!(f, "&gt;")?;
            } else {
                write!(f, ">")?;
            }
        } else {
            fmt_type(type_, f, use_absolute, cx)?;
        }
        Ok(())
    }
}

impl clean::Arguments {
    pub(crate) fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| {
            self.values
                .iter()
                .map(|input| {
                    fmt::from_fn(|f| {
                        if !input.name.is_empty() {
                            write!(f, "{}: ", input.name)?;
                        }
                        input.type_.print(cx).fmt(f)
                    })
                })
                .joined(", ", f)
        })
    }
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
struct Indent(usize);

impl Display for Indent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (0..self.0).for_each(|_| {
            f.write_char(' ').unwrap();
        });
        Ok(())
    }
}

impl clean::FnDecl {
    pub(crate) fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| {
            let ellipsis = if self.c_variadic { ", ..." } else { "" };
            if f.alternate() {
                write!(
                    f,
                    "({args:#}{ellipsis}){arrow:#}",
                    args = self.inputs.print(cx),
                    ellipsis = ellipsis,
                    arrow = self.print_output(cx)
                )
            } else {
                write!(
                    f,
                    "({args}{ellipsis}){arrow}",
                    args = self.inputs.print(cx),
                    ellipsis = ellipsis,
                    arrow = self.print_output(cx)
                )
            }
        })
    }

    /// * `header_len`: The length of the function header and name. In other words, the number of
    ///   characters in the function declaration up to but not including the parentheses.
    ///   This is expected to go into a `<pre>`/`code-header` block, so indentation and newlines
    ///   are preserved.
    /// * `indent`: The number of spaces to indent each successive line with, if line-wrapping is
    ///   necessary.
    pub(crate) fn full_print(
        &self,
        header_len: usize,
        indent: usize,
        cx: &Context<'_>,
    ) -> impl Display {
        fmt::from_fn(move |f| {
            // First, generate the text form of the declaration, with no line wrapping, and count the bytes.
            let mut counter = WriteCounter(0);
            write!(&mut counter, "{:#}", fmt::from_fn(|f| { self.inner_full_print(None, f, cx) }))
                .unwrap();
            // If the text form was over 80 characters wide, we will line-wrap our output.
            let line_wrapping_indent =
                if header_len + counter.0 > 80 { Some(indent) } else { None };
            // Generate the final output. This happens to accept `{:#}` formatting to get textual
            // output but in practice it is only formatted with `{}` to get HTML output.
            self.inner_full_print(line_wrapping_indent, f, cx)
        })
    }

    fn inner_full_print(
        &self,
        // For None, the declaration will not be line-wrapped. For Some(n),
        // the declaration will be line-wrapped, with an indent of n spaces.
        line_wrapping_indent: Option<usize>,
        f: &mut fmt::Formatter<'_>,
        cx: &Context<'_>,
    ) -> fmt::Result {
        let amp = if f.alternate() { "&" } else { "&amp;" };

        write!(f, "(")?;
        if let Some(n) = line_wrapping_indent
            && !self.inputs.values.is_empty()
        {
            write!(f, "\n{}", Indent(n + 4))?;
        }

        let last_input_index = self.inputs.values.len().checked_sub(1);
        for (i, input) in self.inputs.values.iter().enumerate() {
            if let Some(selfty) = input.to_receiver() {
                match selfty {
                    clean::SelfTy => {
                        write!(f, "self")?;
                    }
                    clean::BorrowedRef { lifetime, mutability, type_: box clean::SelfTy } => {
                        write!(f, "{amp}")?;
                        if let Some(lt) = lifetime {
                            write!(f, "{lt} ", lt = lt.print())?;
                        }
                        write!(f, "{mutability}self", mutability = mutability.print_with_space())?;
                    }
                    _ => {
                        write!(f, "self: ")?;
                        selfty.print(cx).fmt(f)?;
                    }
                }
            } else {
                if input.is_const {
                    write!(f, "const ")?;
                }
                write!(f, "{}: ", input.name)?;
                input.type_.print(cx).fmt(f)?;
            }
            match (line_wrapping_indent, last_input_index) {
                (_, None) => (),
                (None, Some(last_i)) if i != last_i => write!(f, ", ")?,
                (None, Some(_)) => (),
                (Some(n), Some(last_i)) if i != last_i => write!(f, ",\n{}", Indent(n + 4))?,
                (Some(_), Some(_)) => writeln!(f, ",")?,
            }
        }

        if self.c_variadic {
            match line_wrapping_indent {
                None => write!(f, ", ...")?,
                Some(n) => writeln!(f, "{}...", Indent(n + 4))?,
            };
        }

        match line_wrapping_indent {
            None => write!(f, ")")?,
            Some(n) => write!(f, "{})", Indent(n))?,
        };

        self.print_output(cx).fmt(f)
    }

    fn print_output(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| match &self.output {
            clean::Tuple(tys) if tys.is_empty() => Ok(()),
            ty if f.alternate() => {
                write!(f, " -> {:#}", ty.print(cx))
            }
            ty => write!(f, " -&gt; {}", ty.print(cx)),
        })
    }
}

pub(crate) fn visibility_print_with_space(item: &clean::Item, cx: &Context<'_>) -> impl Display {
    use std::fmt::Write as _;
    let vis: Cow<'static, str> = match item.visibility(cx.tcx()) {
        None => "".into(),
        Some(ty::Visibility::Public) => "pub ".into(),
        Some(ty::Visibility::Restricted(vis_did)) => {
            // FIXME(camelid): This may not work correctly if `item_did` is a module.
            //                 However, rustdoc currently never displays a module's
            //                 visibility, so it shouldn't matter.
            let parent_module = find_nearest_parent_module(cx.tcx(), item.item_id.expect_def_id());

            if vis_did.is_crate_root() {
                "pub(crate) ".into()
            } else if parent_module == Some(vis_did) {
                // `pub(in foo)` where `foo` is the parent module
                // is the same as no visibility modifier
                "".into()
            } else if parent_module.and_then(|parent| find_nearest_parent_module(cx.tcx(), parent))
                == Some(vis_did)
            {
                "pub(super) ".into()
            } else {
                let path = cx.tcx().def_path(vis_did);
                debug!("path={path:?}");
                // modified from `resolved_path()` to work with `DefPathData`
                let last_name = path.data.last().unwrap().data.get_opt_name().unwrap();
                let anchor = anchor(vis_did, last_name, cx);

                let mut s = "pub(in ".to_owned();
                for seg in &path.data[..path.data.len() - 1] {
                    let _ = write!(s, "{}::", seg.data.get_opt_name().unwrap());
                }
                let _ = write!(s, "{anchor}) ");
                s.into()
            }
        }
    };

    let is_doc_hidden = item.is_doc_hidden();
    fmt::from_fn(move |f| {
        if is_doc_hidden {
            f.write_str("#[doc(hidden)] ")?;
        }

        f.write_str(&vis)
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

impl clean::Import {
    pub(crate) fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| match self.kind {
            clean::ImportKind::Simple(name) => {
                if name == self.source.path.last() {
                    write!(f, "use {};", self.source.print(cx))
                } else {
                    write!(f, "use {source} as {name};", source = self.source.print(cx))
                }
            }
            clean::ImportKind::Glob => {
                if self.source.path.segments.is_empty() {
                    write!(f, "use *;")
                } else {
                    write!(f, "use {}::*;", self.source.print(cx))
                }
            }
        })
    }
}

impl clean::ImportSource {
    pub(crate) fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| match self.did {
            Some(did) => resolved_path(f, did, &self.path, true, false, cx),
            _ => {
                for seg in &self.path.segments[..self.path.segments.len() - 1] {
                    write!(f, "{}::", seg.name)?;
                }
                let name = self.path.last();
                if let hir::def::Res::PrimTy(p) = self.path.res {
                    primitive_link(f, PrimitiveType::from(p), format_args!("{name}"), cx)?;
                } else {
                    f.write_str(name.as_str())?;
                }
                Ok(())
            }
        })
    }
}

impl clean::AssocItemConstraint {
    pub(crate) fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| {
            f.write_str(self.assoc.name.as_str())?;
            self.assoc.args.print(cx).fmt(f)?;
            match self.kind {
                clean::AssocItemConstraintKind::Equality { ref term } => {
                    f.write_str(" = ")?;
                    term.print(cx).fmt(f)?;
                }
                clean::AssocItemConstraintKind::Bound { ref bounds } => {
                    if !bounds.is_empty() {
                        f.write_str(": ")?;
                        print_generic_bounds(bounds, cx).fmt(f)?;
                    }
                }
            }
            Ok(())
        })
    }
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

pub(crate) fn print_default_space(v: bool) -> &'static str {
    if v { "default " } else { "" }
}

impl clean::GenericArg {
    pub(crate) fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| match self {
            clean::GenericArg::Lifetime(lt) => lt.print().fmt(f),
            clean::GenericArg::Type(ty) => ty.print(cx).fmt(f),
            clean::GenericArg::Const(ct) => ct.print(cx.tcx()).fmt(f),
            clean::GenericArg::Infer => Display::fmt("_", f),
        })
    }
}

impl clean::Term {
    pub(crate) fn print(&self, cx: &Context<'_>) -> impl Display {
        fmt::from_fn(move |f| match self {
            clean::Term::Type(ty) => ty.print(cx).fmt(f),
            clean::Term::Constant(ct) => ct.print(cx.tcx()).fmt(f),
        })
    }
}
