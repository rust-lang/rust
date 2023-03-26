//! HTML formatting module
//!
//! This module contains a large number of `fmt::Display` implementations for
//! various types in `rustdoc::clean`.
//!
//! These implementations all emit HTML. As an internal implementation detail,
//! some of them support an alternate format that emits text, but that should
//! not be used external to this module.

use std::borrow::Cow;
use std::cell::Cell;
use std::fmt::{self, Write};
use std::iter::{self, once};

use rustc_ast as ast;
use rustc_attr::{ConstStability, StabilityLevel};
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_metadata::creader::{CStore, LoadedMacro};
use rustc_middle::ty;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::kw;
use rustc_span::{sym, Symbol};
use rustc_target::spec::abi::Abi;

use itertools::Itertools;

use crate::clean::{
    self, types::ExternalLocation, utils::find_nearest_parent_module, ExternalCrate, ItemId,
    PrimitiveType,
};
use crate::formats::item_type::ItemType;
use crate::html::escape::Escape;
use crate::html::render::Context;
use crate::passes::collect_intra_doc_links::UrlFragment;

use super::url_parts_builder::estimate_item_path_byte_length;
use super::url_parts_builder::UrlPartsBuilder;

pub(crate) trait Print {
    fn print(self, buffer: &mut Buffer);
}

impl<F> Print for F
where
    F: FnOnce(&mut Buffer),
{
    fn print(self, buffer: &mut Buffer) {
        (self)(buffer)
    }
}

impl Print for String {
    fn print(self, buffer: &mut Buffer) {
        buffer.write_str(&self);
    }
}

impl Print for &'_ str {
    fn print(self, buffer: &mut Buffer) {
        buffer.write_str(self);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Buffer {
    for_html: bool,
    buffer: String,
}

impl core::fmt::Write for Buffer {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.buffer.write_str(s)
    }

    #[inline]
    fn write_char(&mut self, c: char) -> fmt::Result {
        self.buffer.write_char(c)
    }

    #[inline]
    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> fmt::Result {
        self.buffer.write_fmt(args)
    }
}

impl Buffer {
    pub(crate) fn empty_from(v: &Buffer) -> Buffer {
        Buffer { for_html: v.for_html, buffer: String::new() }
    }

    pub(crate) fn html() -> Buffer {
        Buffer { for_html: true, buffer: String::new() }
    }

    pub(crate) fn new() -> Buffer {
        Buffer { for_html: false, buffer: String::new() }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub(crate) fn into_inner(self) -> String {
        self.buffer
    }

    pub(crate) fn push_str(&mut self, s: &str) {
        self.buffer.push_str(s);
    }

    pub(crate) fn push_buffer(&mut self, other: Buffer) {
        self.buffer.push_str(&other.buffer);
    }

    // Intended for consumption by write! and writeln! (std::fmt) but without
    // the fmt::Result return type imposed by fmt::Write (and avoiding the trait
    // import).
    pub(crate) fn write_str(&mut self, s: &str) {
        self.buffer.push_str(s);
    }

    // Intended for consumption by write! and writeln! (std::fmt) but without
    // the fmt::Result return type imposed by fmt::Write (and avoiding the trait
    // import).
    pub(crate) fn write_fmt(&mut self, v: fmt::Arguments<'_>) {
        self.buffer.write_fmt(v).unwrap();
    }

    pub(crate) fn to_display<T: Print>(mut self, t: T) -> String {
        t.print(&mut self);
        self.into_inner()
    }

    pub(crate) fn is_for_html(&self) -> bool {
        self.for_html
    }

    pub(crate) fn reserve(&mut self, additional: usize) {
        self.buffer.reserve(additional)
    }

    pub(crate) fn len(&self) -> usize {
        self.buffer.len()
    }
}

pub(crate) fn comma_sep<T: fmt::Display>(
    items: impl Iterator<Item = T>,
    space_after_comma: bool,
) -> impl fmt::Display {
    display_fn(move |f| {
        for (i, item) in items.enumerate() {
            if i != 0 {
                write!(f, ",{}", if space_after_comma { " " } else { "" })?;
            }
            fmt::Display::fmt(&item, f)?;
        }
        Ok(())
    })
}

pub(crate) fn print_generic_bounds<'a, 'tcx: 'a>(
    bounds: &'a [clean::GenericBound],
    cx: &'a Context<'tcx>,
) -> impl fmt::Display + 'a + Captures<'tcx> {
    display_fn(move |f| {
        let mut bounds_dup = FxHashSet::default();

        for (i, bound) in bounds.iter().filter(|b| bounds_dup.insert(b.clone())).enumerate() {
            if i > 0 {
                f.write_str(" + ")?;
            }
            fmt::Display::fmt(&bound.print(cx), f)?;
        }
        Ok(())
    })
}

impl clean::GenericParamDef {
    pub(crate) fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| match &self.kind {
            clean::GenericParamDefKind::Lifetime { outlives } => {
                write!(f, "{}", self.name)?;

                if !outlives.is_empty() {
                    f.write_str(": ")?;
                    for (i, lt) in outlives.iter().enumerate() {
                        if i != 0 {
                            f.write_str(" + ")?;
                        }
                        write!(f, "{}", lt.print())?;
                    }
                }

                Ok(())
            }
            clean::GenericParamDefKind::Type { bounds, default, .. } => {
                f.write_str(self.name.as_str())?;

                if !bounds.is_empty() {
                    if f.alternate() {
                        write!(f, ": {:#}", print_generic_bounds(bounds, cx))?;
                    } else {
                        write!(f, ": {}", print_generic_bounds(bounds, cx))?;
                    }
                }

                if let Some(ref ty) = default {
                    if f.alternate() {
                        write!(f, " = {:#}", ty.print(cx))?;
                    } else {
                        write!(f, " = {}", ty.print(cx))?;
                    }
                }

                Ok(())
            }
            clean::GenericParamDefKind::Const { ty, default, .. } => {
                if f.alternate() {
                    write!(f, "const {}: {:#}", self.name, ty.print(cx))?;
                } else {
                    write!(f, "const {}: {}", self.name, ty.print(cx))?;
                }

                if let Some(default) = default {
                    if f.alternate() {
                        write!(f, " = {:#}", default)?;
                    } else {
                        write!(f, " = {}", default)?;
                    }
                }

                Ok(())
            }
        })
    }
}

impl clean::Generics {
    pub(crate) fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| {
            let mut real_params =
                self.params.iter().filter(|p| !p.is_synthetic_type_param()).peekable();
            if real_params.peek().is_none() {
                return Ok(());
            }

            if f.alternate() {
                write!(f, "<{:#}>", comma_sep(real_params.map(|g| g.print(cx)), true))
            } else {
                write!(f, "&lt;{}&gt;", comma_sep(real_params.map(|g| g.print(cx)), true))
            }
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ending {
    Newline,
    NoNewline,
}

/// * The Generics from which to emit a where-clause.
/// * The number of spaces to indent each line with.
/// * Whether the where-clause needs to add a comma and newline after the last bound.
pub(crate) fn print_where_clause<'a, 'tcx: 'a>(
    gens: &'a clean::Generics,
    cx: &'a Context<'tcx>,
    indent: usize,
    ending: Ending,
) -> impl fmt::Display + 'a + Captures<'tcx> {
    display_fn(move |f| {
        let mut where_predicates = gens.where_predicates.iter().filter(|pred| {
            !matches!(pred, clean::WherePredicate::BoundPredicate { bounds, .. } if bounds.is_empty())
        }).map(|pred| {
            display_fn(move |f| {
                if f.alternate() {
                    f.write_str(" ")?;
                } else {
                    f.write_str("\n")?;
                }

                match pred {
                    clean::WherePredicate::BoundPredicate { ty, bounds, bound_params } => {
                        let ty_cx = ty.print(cx);
                        let generic_bounds = print_generic_bounds(bounds, cx);

                        if bound_params.is_empty() {
                            if f.alternate() {
                                write!(f, "{ty_cx:#}: {generic_bounds:#}")
                            } else {
                                write!(f, "{ty_cx}: {generic_bounds}")
                            }
                        } else {
                            if f.alternate() {
                                write!(
                                    f,
                                    "for<{:#}> {ty_cx:#}: {generic_bounds:#}",
                                    comma_sep(bound_params.iter().map(|lt| lt.print()), true)
                                )
                            } else {
                                write!(
                                    f,
                                    "for&lt;{}&gt; {ty_cx}: {generic_bounds}",
                                    comma_sep(bound_params.iter().map(|lt| lt.print()), true)
                                )
                            }
                        }
                    }
                    clean::WherePredicate::RegionPredicate { lifetime, bounds } => {
                        let mut bounds_display = String::new();
                        for bound in bounds.iter().map(|b| b.print(cx)) {
                            write!(bounds_display, "{bound} + ")?;
                        }
                        bounds_display.truncate(bounds_display.len() - " + ".len());
                        write!(f, "{}: {bounds_display}", lifetime.print())
                    }
                    // FIXME(fmease): Render bound params.
                    clean::WherePredicate::EqPredicate { lhs, rhs, bound_params: _ } => {
                        if f.alternate() {
                            write!(f, "{:#} == {:#}", lhs.print(cx), rhs.print(cx))
                        } else {
                            write!(f, "{} == {}", lhs.print(cx), rhs.print(cx))
                        }
                    }
                }
            })
        }).peekable();

        if where_predicates.peek().is_none() {
            return Ok(());
        }

        let where_preds = comma_sep(where_predicates, false);
        let clause = if f.alternate() {
            if ending == Ending::Newline {
                format!(" where{where_preds},")
            } else {
                format!(" where{where_preds}")
            }
        } else {
            let mut br_with_padding = String::with_capacity(6 * indent + 28);
            br_with_padding.push_str("\n");

            let padding_amout =
                if ending == Ending::Newline { indent + 4 } else { indent + "fn where ".len() };

            for _ in 0..padding_amout {
                br_with_padding.push_str(" ");
            }
            let where_preds = where_preds.to_string().replace('\n', &br_with_padding);

            if ending == Ending::Newline {
                let mut clause = " ".repeat(indent.saturating_sub(1));
                write!(clause, "<span class=\"where fmt-newline\">where{where_preds},</span>")?;
                clause
            } else {
                // insert a newline after a single space but before multiple spaces at the start
                if indent == 0 {
                    format!("\n<span class=\"where\">where{where_preds}</span>")
                } else {
                    // put the first one on the same line as the 'where' keyword
                    let where_preds = where_preds.replacen(&br_with_padding, " ", 1);

                    let mut clause = br_with_padding;
                    clause.truncate(clause.len() - "where ".len());

                    write!(clause, "<span class=\"where\">where{where_preds}</span>")?;
                    clause
                }
            }
        };
        write!(f, "{clause}")
    })
}

impl clean::Lifetime {
    pub(crate) fn print(&self) -> impl fmt::Display + '_ {
        self.0.as_str()
    }
}

impl clean::Constant {
    pub(crate) fn print(&self, tcx: TyCtxt<'_>) -> impl fmt::Display + '_ {
        let expr = self.expr(tcx);
        display_fn(
            move |f| {
                if f.alternate() { f.write_str(&expr) } else { write!(f, "{}", Escape(&expr)) }
            },
        )
    }
}

impl clean::PolyTrait {
    fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| {
            if !self.generic_params.is_empty() {
                if f.alternate() {
                    write!(
                        f,
                        "for<{:#}> ",
                        comma_sep(self.generic_params.iter().map(|g| g.print(cx)), true)
                    )?;
                } else {
                    write!(
                        f,
                        "for&lt;{}&gt; ",
                        comma_sep(self.generic_params.iter().map(|g| g.print(cx)), true)
                    )?;
                }
            }
            if f.alternate() {
                write!(f, "{:#}", self.trait_.print(cx))
            } else {
                write!(f, "{}", self.trait_.print(cx))
            }
        })
    }
}

impl clean::GenericBound {
    pub(crate) fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| match self {
            clean::GenericBound::Outlives(lt) => write!(f, "{}", lt.print()),
            clean::GenericBound::TraitBound(ty, modifier) => {
                let modifier_str = match modifier {
                    hir::TraitBoundModifier::None => "",
                    hir::TraitBoundModifier::Maybe => "?",
                    // ~const is experimental; do not display those bounds in rustdoc
                    hir::TraitBoundModifier::MaybeConst => "",
                };
                if f.alternate() {
                    write!(f, "{}{:#}", modifier_str, ty.print(cx))
                } else {
                    write!(f, "{}{}", modifier_str, ty.print(cx))
                }
            }
        })
    }
}

impl clean::GenericArgs {
    fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| {
            match self {
                clean::GenericArgs::AngleBracketed { args, bindings } => {
                    if !args.is_empty() || !bindings.is_empty() {
                        if f.alternate() {
                            f.write_str("<")?;
                        } else {
                            f.write_str("&lt;")?;
                        }
                        let mut comma = false;
                        for arg in args.iter() {
                            if comma {
                                f.write_str(", ")?;
                            }
                            comma = true;
                            if f.alternate() {
                                write!(f, "{:#}", arg.print(cx))?;
                            } else {
                                write!(f, "{}", arg.print(cx))?;
                            }
                        }
                        for binding in bindings.iter() {
                            if comma {
                                f.write_str(", ")?;
                            }
                            comma = true;
                            if f.alternate() {
                                write!(f, "{:#}", binding.print(cx))?;
                            } else {
                                write!(f, "{}", binding.print(cx))?;
                            }
                        }
                        if f.alternate() {
                            f.write_str(">")?;
                        } else {
                            f.write_str("&gt;")?;
                        }
                    }
                }
                clean::GenericArgs::Parenthesized { inputs, output } => {
                    f.write_str("(")?;
                    let mut comma = false;
                    for ty in inputs.iter() {
                        if comma {
                            f.write_str(", ")?;
                        }
                        comma = true;
                        if f.alternate() {
                            write!(f, "{:#}", ty.print(cx))?;
                        } else {
                            write!(f, "{}", ty.print(cx))?;
                        }
                    }
                    f.write_str(")")?;
                    if let Some(ref ty) = *output {
                        if f.alternate() {
                            write!(f, " -> {:#}", ty.print(cx))?;
                        } else {
                            write!(f, " -&gt; {}", ty.print(cx))?;
                        }
                    }
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
    let tcx = cx.shared.tcx;
    let crate_name = tcx.crate_name(def_id.krate);
    let cache = cx.cache();

    let fqp: Vec<Symbol> = tcx
        .def_path(def_id)
        .data
        .into_iter()
        .filter_map(|elem| {
            // extern blocks (and a few others things) have an empty name.
            match elem.data.get_opt_name() {
                Some(s) if !s.is_empty() => Some(s),
                _ => None,
            }
        })
        .collect();
    let mut relative = fqp.iter().copied();
    let cstore = CStore::from_tcx(tcx);
    // We need this to prevent a `panic` when this function is used from intra doc links...
    if !cstore.has_crate_data(def_id.krate) {
        debug!("No data for crate {}", crate_name);
        return Err(HrefError::NotInExternalCache);
    }
    // Check to see if it is a macro 2.0 or built-in macro.
    // More information in <https://rust-lang.github.io/rfcs/1584-macros.html>.
    let is_macro_2 = match cstore.load_macro_untracked(def_id, tcx.sess) {
        LoadedMacro::MacroDef(def, _) => {
            // If `ast_def.macro_rules` is `true`, then it's not a macro 2.0.
            matches!(&def.kind, ast::ItemKind::MacroDef(ast_def) if !ast_def.macro_rules)
        }
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
        *last = Symbol::intern(&format!("macro.{}.html", last.as_str()));
    }

    let url = match cache.extern_locations[&def_id.krate] {
        ExternalLocation::Remote(ref s) => {
            // `ExternalLocation::Remote` always end with a `/`.
            format!("{}{}", s, path.iter().map(|p| p.as_str()).join("/"))
        }
        ExternalLocation::Local => {
            // `root_path` always end with a `/`.
            format!(
                "{}{}/{}",
                root_path.unwrap_or(""),
                crate_name,
                path.iter().map(|p| p.as_str()).join("/")
            )
        }
        ExternalLocation::Unknown => {
            debug!("crate {} not in cache when linkifying macros", crate_name);
            return Err(HrefError::NotInExternalCache);
        }
    };
    Ok((url, ItemType::Macro, fqp))
}

pub(crate) fn href_with_root_path(
    did: DefId,
    cx: &Context<'_>,
    root_path: Option<&str>,
) -> Result<(String, ItemType, Vec<Symbol>), HrefError> {
    let tcx = cx.tcx();
    let def_kind = tcx.def_kind(did);
    let did = match def_kind {
        DefKind::AssocTy | DefKind::AssocFn | DefKind::AssocConst | DefKind::Variant => {
            // documented on their parent's page
            tcx.parent(did)
        }
        _ => did,
    };
    let cache = cx.cache();
    let relative_to = &cx.current;
    fn to_module_fqp(shortty: ItemType, fqp: &[Symbol]) -> &[Symbol] {
        if shortty == ItemType::Module { fqp } else { &fqp[..fqp.len() - 1] }
    }

    if !did.is_local()
        && !cache.effective_visibilities.is_directly_public(tcx, did)
        && !cache.document_private
        && !cache.primitive_locations.values().any(|&id| id == did)
    {
        return Err(HrefError::Private);
    }

    let mut is_remote = false;
    let (fqp, shortty, mut url_parts) = match cache.paths.get(&did) {
        Some(&(ref fqp, shortty)) => (fqp, shortty, {
            let module_fqp = to_module_fqp(shortty, fqp.as_slice());
            debug!(?fqp, ?shortty, ?module_fqp);
            href_relative_parts(module_fqp, relative_to).collect()
        }),
        None => {
            if let Some(&(ref fqp, shortty)) = cache.external_paths.get(&did) {
                let module_fqp = to_module_fqp(shortty, fqp);
                (
                    fqp,
                    shortty,
                    match cache.extern_locations[&did.krate] {
                        ExternalLocation::Remote(ref s) => {
                            is_remote = true;
                            let s = s.trim_end_matches('/');
                            let mut builder = UrlPartsBuilder::singleton(s);
                            builder.extend(module_fqp.iter().copied());
                            builder
                        }
                        ExternalLocation::Local => {
                            href_relative_parts(module_fqp, relative_to).collect()
                        }
                        ExternalLocation::Unknown => return Err(HrefError::DocumentationNotBuilt),
                    },
                )
            } else if matches!(def_kind, DefKind::Macro(_)) {
                return generate_macro_def_id_path(did, cx, root_path);
            } else {
                return Err(HrefError::NotInExternalCache);
            }
        }
    };
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
            let prefix = shortty.as_str();
            let last = fqp.last().unwrap();
            url_parts.push_fmt(format_args!("{}.{}.html", prefix, last));
        }
    }
    Ok((url_parts.finish(), shortty, fqp.to_vec()))
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
            let fqp_module = &fqp[i..fqp.len()];
            return Box::new(
                iter::repeat(sym::dotdot)
                    .take(dissimilar_part_count)
                    .chain(fqp_module.iter().copied()),
            );
        }
    }
    // e.g. linking to std::sync::atomic from std::sync
    if relative_to_fqp.len() < fqp.len() {
        Box::new(fqp[relative_to_fqp.len()..fqp.len()].iter().copied())
    // e.g. linking to std::sync from std::sync::atomic
    } else if fqp.len() < relative_to_fqp.len() {
        let dissimilar_part_count = relative_to_fqp.len() - fqp.len();
        Box::new(iter::repeat(sym::dotdot).take(dissimilar_part_count))
    // linking to the same module
    } else {
        Box::new(iter::empty())
    }
}

pub(crate) fn link_tooltip(did: DefId, fragment: &Option<UrlFragment>, cx: &Context<'_>) -> String {
    let cache = cx.cache();
    let Some((fqp, shortty)) = cache.paths.get(&did)
        .or_else(|| cache.external_paths.get(&did))
        else { return String::new() };
    let mut buf = Buffer::new();
    let fqp = if *shortty == ItemType::Primitive {
        // primitives are documented in a crate, but not actually part of it
        &fqp[fqp.len() - 1..]
    } else {
        &fqp
    };
    if let &Some(UrlFragment::Item(id)) = fragment {
        write!(buf, "{} ", cx.tcx().def_descr(id));
        for component in fqp {
            write!(buf, "{component}::");
        }
        write!(buf, "{}", cx.tcx().item_name(id));
    } else if !fqp.is_empty() {
        let mut fqp_it = fqp.into_iter();
        write!(buf, "{shortty} {}", fqp_it.next().unwrap());
        for component in fqp_it {
            write!(buf, "::{component}");
        }
    }
    buf.into_inner()
}

/// Used to render a [`clean::Path`].
fn resolved_path<'cx>(
    w: &mut fmt::Formatter<'_>,
    did: DefId,
    path: &clean::Path,
    print_all: bool,
    use_absolute: bool,
    cx: &'cx Context<'_>,
) -> fmt::Result {
    let last = path.segments.last().unwrap();

    if print_all {
        for seg in &path.segments[..path.segments.len() - 1] {
            write!(w, "{}::", if seg.name == kw::PathRoot { "" } else { seg.name.as_str() })?;
        }
    }
    if w.alternate() {
        write!(w, "{}{:#}", &last.name, last.args.print(cx))?;
    } else {
        let path = if use_absolute {
            if let Ok((_, _, fqp)) = href(did, cx) {
                format!(
                    "{}::{}",
                    join_with_double_colon(&fqp[..fqp.len() - 1]),
                    anchor(did, *fqp.last().unwrap(), cx)
                )
            } else {
                last.name.to_string()
            }
        } else {
            anchor(did, last.name, cx).to_string()
        };
        write!(w, "{}{}", path, last.args.print(cx))?;
    }
    Ok(())
}

fn primitive_link(
    f: &mut fmt::Formatter<'_>,
    prim: clean::PrimitiveType,
    name: &str,
    cx: &Context<'_>,
) -> fmt::Result {
    primitive_link_fragment(f, prim, name, "", cx)
}

fn primitive_link_fragment(
    f: &mut fmt::Formatter<'_>,
    prim: clean::PrimitiveType,
    name: &str,
    fragment: &str,
    cx: &Context<'_>,
) -> fmt::Result {
    let m = &cx.cache();
    let mut needs_termination = false;
    if !f.alternate() {
        match m.primitive_locations.get(&prim) {
            Some(&def_id) if def_id.is_local() => {
                let len = cx.current.len();
                let len = if len == 0 { 0 } else { len - 1 };
                write!(
                    f,
                    "<a class=\"primitive\" href=\"{}primitive.{}.html{fragment}\">",
                    "../".repeat(len),
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
                            iter::repeat(sym::dotdot).take(cx.current.len() - 1).collect()
                        } else {
                            iter::repeat(sym::dotdot)
                                .take(cx.current.len())
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
    write!(f, "{}", name)?;
    if needs_termination {
        write!(f, "</a>")?;
    }
    Ok(())
}

/// Helper to render type parameters
fn tybounds<'a, 'tcx: 'a>(
    bounds: &'a [clean::PolyTrait],
    lt: &'a Option<clean::Lifetime>,
    cx: &'a Context<'tcx>,
) -> impl fmt::Display + 'a + Captures<'tcx> {
    display_fn(move |f| {
        for (i, bound) in bounds.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }

            fmt::Display::fmt(&bound.print(cx), f)?;
        }

        if let Some(lt) = lt {
            write!(f, " + ")?;
            fmt::Display::fmt(&lt.print(), f)?;
        }
        Ok(())
    })
}

pub(crate) fn anchor<'a, 'cx: 'a>(
    did: DefId,
    text: Symbol,
    cx: &'cx Context<'_>,
) -> impl fmt::Display + 'a {
    let parts = href(did, cx);
    display_fn(move |f| {
        if let Ok((url, short_ty, fqp)) = parts {
            write!(
                f,
                r#"<a class="{}" href="{}" title="{} {}">{}</a>"#,
                short_ty,
                url,
                short_ty,
                join_with_double_colon(&fqp),
                text.as_str()
            )
        } else {
            write!(f, "{}", text)
        }
    })
}

fn fmt_type<'cx>(
    t: &clean::Type,
    f: &mut fmt::Formatter<'_>,
    use_absolute: bool,
    cx: &'cx Context<'_>,
) -> fmt::Result {
    trace!("fmt_type(t = {:?})", t);

    match *t {
        clean::Generic(name) => write!(f, "{}", name),
        clean::Type::Path { ref path } => {
            // Paths like `T::Output` and `Self::Output` should be rendered with all segments.
            let did = path.def_id();
            resolved_path(f, did, path, path.is_assoc_ty(), use_absolute, cx)
        }
        clean::DynTrait(ref bounds, ref lt) => {
            f.write_str("dyn ")?;
            fmt::Display::fmt(&tybounds(bounds, lt, cx), f)
        }
        clean::Infer => write!(f, "_"),
        clean::Primitive(clean::PrimitiveType::Never) => {
            primitive_link(f, PrimitiveType::Never, "!", cx)
        }
        clean::Primitive(prim) => primitive_link(f, prim, prim.as_sym().as_str(), cx),
        clean::BareFunction(ref decl) => {
            if f.alternate() {
                write!(
                    f,
                    "{:#}{}{:#}fn{:#}",
                    decl.print_hrtb_with_space(cx),
                    decl.unsafety.print_with_space(),
                    print_abi_with_space(decl.abi),
                    decl.decl.print(cx),
                )
            } else {
                write!(
                    f,
                    "{}{}{}",
                    decl.print_hrtb_with_space(cx),
                    decl.unsafety.print_with_space(),
                    print_abi_with_space(decl.abi)
                )?;
                primitive_link(f, PrimitiveType::Fn, "fn", cx)?;
                write!(f, "{}", decl.decl.print(cx))
            }
        }
        clean::Tuple(ref typs) => {
            match &typs[..] {
                &[] => primitive_link(f, PrimitiveType::Unit, "()", cx),
                [one] => {
                    if let clean::Generic(name) = one {
                        primitive_link(f, PrimitiveType::Tuple, &format!("({name},)"), cx)
                    } else {
                        write!(f, "(")?;
                        // Carry `f.alternate()` into this display w/o branching manually.
                        fmt::Display::fmt(&one.print(cx), f)?;
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
                            &format!("({})", generic_names.iter().map(|s| s.as_str()).join(", ")),
                            cx,
                        )
                    } else {
                        write!(f, "(")?;
                        for (i, item) in many.iter().enumerate() {
                            if i != 0 {
                                write!(f, ", ")?;
                            }
                            // Carry `f.alternate()` into this display w/o branching manually.
                            fmt::Display::fmt(&item.print(cx), f)?;
                        }
                        write!(f, ")")
                    }
                }
            }
        }
        clean::Slice(ref t) => match **t {
            clean::Generic(name) => {
                primitive_link(f, PrimitiveType::Slice, &format!("[{name}]"), cx)
            }
            _ => {
                write!(f, "[")?;
                fmt::Display::fmt(&t.print(cx), f)?;
                write!(f, "]")
            }
        },
        clean::Array(ref t, ref n) => match **t {
            clean::Generic(name) if !f.alternate() => primitive_link(
                f,
                PrimitiveType::Array,
                &format!("[{name}; {n}]", n = Escape(n)),
                cx,
            ),
            _ => {
                write!(f, "[")?;
                fmt::Display::fmt(&t.print(cx), f)?;
                if f.alternate() {
                    write!(f, "; {n}")?;
                } else {
                    write!(f, "; ")?;
                    primitive_link(f, PrimitiveType::Array, &format!("{n}", n = Escape(n)), cx)?;
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
                let text = if f.alternate() {
                    format!("*{} {:#}", m, t.print(cx))
                } else {
                    format!("*{} {}", m, t.print(cx))
                };
                primitive_link(f, clean::PrimitiveType::RawPointer, &text, cx)
            } else {
                primitive_link(f, clean::PrimitiveType::RawPointer, &format!("*{} ", m), cx)?;
                fmt::Display::fmt(&t.print(cx), f)
            }
        }
        clean::BorrowedRef { lifetime: ref l, mutability, type_: ref ty } => {
            let lt = match l {
                Some(l) => format!("{} ", l.print()),
                _ => String::new(),
            };
            let m = mutability.print_with_space();
            let amp = if f.alternate() { "&" } else { "&amp;" };
            match **ty {
                clean::DynTrait(ref bounds, ref trait_lt)
                    if bounds.len() > 1 || trait_lt.is_some() =>
                {
                    write!(f, "{}{}{}(", amp, lt, m)?;
                    fmt_type(ty, f, use_absolute, cx)?;
                    write!(f, ")")
                }
                clean::Generic(name) => {
                    primitive_link(f, PrimitiveType::Reference, &format!("{amp}{lt}{m}{name}"), cx)
                }
                _ => {
                    write!(f, "{}{}{}", amp, lt, m)?;
                    fmt_type(ty, f, use_absolute, cx)
                }
            }
        }
        clean::ImplTrait(ref bounds) => {
            if f.alternate() {
                write!(f, "impl {:#}", print_generic_bounds(bounds, cx))
            } else {
                write!(f, "impl {}", print_generic_bounds(bounds, cx))
            }
        }
        clean::QPath(box clean::QPathData {
            ref assoc,
            ref self_type,
            ref trait_,
            should_show_cast,
        }) => {
            if f.alternate() {
                if should_show_cast {
                    write!(f, "<{:#} as {:#}>::", self_type.print(cx), trait_.print(cx))?
                } else {
                    write!(f, "{:#}::", self_type.print(cx))?
                }
            } else {
                if should_show_cast {
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
            match href(trait_.def_id(), cx) {
                Ok((ref url, _, ref path)) if !f.alternate() => {
                    write!(
                        f,
                        "<a class=\"associatedtype\" href=\"{url}#{shortty}.{name}\" \
                                    title=\"type {path}::{name}\">{name}</a>{args}",
                        url = url,
                        shortty = ItemType::AssocType,
                        name = assoc.name,
                        path = join_with_double_colon(path),
                        args = assoc.args.print(cx),
                    )?;
                }
                _ => write!(f, "{}{:#}", assoc.name, assoc.args.print(cx))?,
            }
            Ok(())
        }
    }
}

impl clean::Type {
    pub(crate) fn print<'b, 'a: 'b, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'b + Captures<'tcx> {
        display_fn(move |f| fmt_type(self, f, false, cx))
    }
}

impl clean::Path {
    pub(crate) fn print<'b, 'a: 'b, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'b + Captures<'tcx> {
        display_fn(move |f| resolved_path(f, self.def_id(), self, false, false, cx))
    }
}

impl clean::Impl {
    pub(crate) fn print<'a, 'tcx: 'a>(
        &'a self,
        use_absolute: bool,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| {
            if f.alternate() {
                write!(f, "impl{:#} ", self.generics.print(cx))?;
            } else {
                write!(f, "impl{} ", self.generics.print(cx))?;
            }

            if let Some(ref ty) = self.trait_ {
                match self.polarity {
                    ty::ImplPolarity::Positive | ty::ImplPolarity::Reservation => {}
                    ty::ImplPolarity::Negative => write!(f, "!")?,
                }
                fmt::Display::fmt(&ty.print(cx), f)?;
                write!(f, " for ")?;
            }

            if let clean::Type::Tuple(types) = &self.for_ &&
                let [clean::Type::Generic(name)] = &types[..] &&
                (self.kind.is_fake_variadic() || self.kind.is_auto())
            {
                // Hardcoded anchor library/core/src/primitive_docs.rs
                // Link should match `# Trait implementations`
                primitive_link_fragment(f, PrimitiveType::Tuple, &format!("({name}₁, {name}₂, …, {name}ₙ)"), "#trait-implementations-1", cx)?;
            } else if let clean::BareFunction(bare_fn) = &self.for_ &&
                let [clean::Argument { type_: clean::Type::Generic(name), .. }] = &bare_fn.decl.inputs.values[..] &&
                (self.kind.is_fake_variadic() || self.kind.is_auto())
            {
                // Hardcoded anchor library/core/src/primitive_docs.rs
                // Link should match `# Trait implementations`

                let hrtb = bare_fn.print_hrtb_with_space(cx);
                let unsafety = bare_fn.unsafety.print_with_space();
                let abi = print_abi_with_space(bare_fn.abi);
                if f.alternate() {
                    write!(
                        f,
                        "{hrtb:#}{unsafety}{abi:#}",
                    )?;
                } else {
                    write!(
                        f,
                        "{hrtb}{unsafety}{abi}",
                    )?;
                }
                let ellipsis = if bare_fn.decl.c_variadic {
                    ", ..."
                } else {
                    ""
                };
                primitive_link_fragment(f, PrimitiveType::Tuple, &format!("fn ({name}₁, {name}₂, …, {name}ₙ{ellipsis})"), "#trait-implementations-1", cx)?;
                // Write output.
                if let clean::FnRetTy::Return(ty) = &bare_fn.decl.output {
                    write!(f, " -> ")?;
                    fmt_type(ty, f, use_absolute, cx)?;
                }
            } else if let Some(ty) = self.kind.as_blanket_ty() {
                fmt_type(ty, f, use_absolute, cx)?;
            } else {
                fmt_type(&self.for_, f, use_absolute, cx)?;
            }

            fmt::Display::fmt(&print_where_clause(&self.generics, cx, 0, Ending::Newline), f)?;
            Ok(())
        })
    }
}

impl clean::Arguments {
    pub(crate) fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| {
            for (i, input) in self.values.iter().enumerate() {
                write!(f, "{}: ", input.name)?;

                if f.alternate() {
                    write!(f, "{:#}", input.type_.print(cx))?;
                } else {
                    write!(f, "{}", input.type_.print(cx))?;
                }
                if i + 1 < self.values.len() {
                    write!(f, ", ")?;
                }
            }
            Ok(())
        })
    }
}

impl clean::FnRetTy {
    pub(crate) fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| match self {
            clean::Return(clean::Tuple(tys)) if tys.is_empty() => Ok(()),
            clean::Return(ty) if f.alternate() => {
                write!(f, " -> {:#}", ty.print(cx))
            }
            clean::Return(ty) => write!(f, " -&gt; {}", ty.print(cx)),
            clean::DefaultReturn => Ok(()),
        })
    }
}

impl clean::BareFunctionDecl {
    fn print_hrtb_with_space<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| {
            if !self.generic_params.is_empty() {
                write!(
                    f,
                    "for&lt;{}&gt; ",
                    comma_sep(self.generic_params.iter().map(|g| g.print(cx)), true)
                )
            } else {
                Ok(())
            }
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

impl fmt::Display for Indent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (0..self.0).for_each(|_| {
            f.write_char(' ').unwrap();
        });
        Ok(())
    }
}

impl clean::FnDecl {
    pub(crate) fn print<'b, 'a: 'b, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'b + Captures<'tcx> {
        display_fn(move |f| {
            let ellipsis = if self.c_variadic { ", ..." } else { "" };
            if f.alternate() {
                write!(
                    f,
                    "({args:#}{ellipsis}){arrow:#}",
                    args = self.inputs.print(cx),
                    ellipsis = ellipsis,
                    arrow = self.output.print(cx)
                )
            } else {
                write!(
                    f,
                    "({args}{ellipsis}){arrow}",
                    args = self.inputs.print(cx),
                    ellipsis = ellipsis,
                    arrow = self.output.print(cx)
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
    pub(crate) fn full_print<'a, 'tcx: 'a>(
        &'a self,
        header_len: usize,
        indent: usize,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| {
            // First, generate the text form of the declaration, with no line wrapping, and count the bytes.
            let mut counter = WriteCounter(0);
            write!(&mut counter, "{:#}", display_fn(|f| { self.inner_full_print(None, f, cx) }))
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
        if let Some(n) = line_wrapping_indent {
            write!(f, "\n{}", Indent(n + 4))?;
        }
        for (i, input) in self.inputs.values.iter().enumerate() {
            if i > 0 {
                match line_wrapping_indent {
                    None => write!(f, ", ")?,
                    Some(n) => write!(f, ",\n{}", Indent(n + 4))?,
                };
            }
            if let Some(selfty) = input.to_self() {
                match selfty {
                    clean::SelfValue => {
                        write!(f, "self")?;
                    }
                    clean::SelfBorrowed(Some(ref lt), mtbl) => {
                        write!(f, "{}{} {}self", amp, lt.print(), mtbl.print_with_space())?;
                    }
                    clean::SelfBorrowed(None, mtbl) => {
                        write!(f, "{}{}self", amp, mtbl.print_with_space())?;
                    }
                    clean::SelfExplicit(ref typ) => {
                        write!(f, "self: ")?;
                        fmt::Display::fmt(&typ.print(cx), f)?;
                    }
                }
            } else {
                if input.is_const {
                    write!(f, "const ")?;
                }
                write!(f, "{}: ", input.name)?;
                fmt::Display::fmt(&input.type_.print(cx), f)?;
            }
        }

        if self.c_variadic {
            match line_wrapping_indent {
                None => write!(f, ", ...")?,
                Some(n) => write!(f, "\n{}...", Indent(n + 4))?,
            };
        }

        match line_wrapping_indent {
            None => write!(f, ")")?,
            Some(n) => write!(f, "\n{})", Indent(n))?,
        };

        fmt::Display::fmt(&self.output.print(cx), f)?;
        Ok(())
    }
}

pub(crate) fn visibility_print_with_space<'a, 'tcx: 'a>(
    visibility: Option<ty::Visibility<DefId>>,
    item_did: ItemId,
    cx: &'a Context<'tcx>,
) -> impl fmt::Display + 'a + Captures<'tcx> {
    use std::fmt::Write as _;

    let to_print: Cow<'static, str> = match visibility {
        None => "".into(),
        Some(ty::Visibility::Public) => "pub ".into(),
        Some(ty::Visibility::Restricted(vis_did)) => {
            // FIXME(camelid): This may not work correctly if `item_did` is a module.
            //                 However, rustdoc currently never displays a module's
            //                 visibility, so it shouldn't matter.
            let parent_module = find_nearest_parent_module(cx.tcx(), item_did.expect_def_id());

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
                debug!("path={:?}", path);
                // modified from `resolved_path()` to work with `DefPathData`
                let last_name = path.data.last().unwrap().data.get_opt_name().unwrap();
                let anchor = anchor(vis_did, last_name, cx);

                let mut s = "pub(in ".to_owned();
                for seg in &path.data[..path.data.len() - 1] {
                    let _ = write!(s, "{}::", seg.data.get_opt_name().unwrap());
                }
                let _ = write!(s, "{}) ", anchor);
                s.into()
            }
        }
    };
    display_fn(move |f| write!(f, "{}", to_print))
}

/// This function is the same as print_with_space, except that it renders no links.
/// It's used for macros' rendered source view, which is syntax highlighted and cannot have
/// any HTML in it.
pub(crate) fn visibility_to_src_with_space<'a, 'tcx: 'a>(
    visibility: Option<ty::Visibility<DefId>>,
    tcx: TyCtxt<'tcx>,
    item_did: DefId,
) -> impl fmt::Display + 'a + Captures<'tcx> {
    let to_print: Cow<'static, str> = match visibility {
        None => "".into(),
        Some(ty::Visibility::Public) => "pub ".into(),
        Some(ty::Visibility::Restricted(vis_did)) => {
            // FIXME(camelid): This may not work correctly if `item_did` is a module.
            //                 However, rustdoc currently never displays a module's
            //                 visibility, so it shouldn't matter.
            let parent_module = find_nearest_parent_module(tcx, item_did);

            if vis_did.is_crate_root() {
                "pub(crate) ".into()
            } else if parent_module == Some(vis_did) {
                // `pub(in foo)` where `foo` is the parent module
                // is the same as no visibility modifier
                "".into()
            } else if parent_module.and_then(|parent| find_nearest_parent_module(tcx, parent))
                == Some(vis_did)
            {
                "pub(super) ".into()
            } else {
                format!("pub(in {}) ", tcx.def_path_str(vis_did)).into()
            }
        }
    };
    display_fn(move |f| f.write_str(&to_print))
}

pub(crate) trait PrintWithSpace {
    fn print_with_space(&self) -> &str;
}

impl PrintWithSpace for hir::Unsafety {
    fn print_with_space(&self) -> &str {
        match self {
            hir::Unsafety::Unsafe => "unsafe ",
            hir::Unsafety::Normal => "",
        }
    }
}

impl PrintWithSpace for hir::IsAsync {
    fn print_with_space(&self) -> &str {
        match self {
            hir::IsAsync::Async => "async ",
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
    s: Option<ConstStability>,
) -> &'static str {
    match (c, s) {
        // const stable or when feature(staged_api) is not set
        (
            hir::Constness::Const,
            Some(ConstStability { level: StabilityLevel::Stable { .. }, .. }),
        )
        | (hir::Constness::Const, None) => "const ",
        // const unstable or not const
        _ => "",
    }
}

impl clean::Import {
    pub(crate) fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| match self.kind {
            clean::ImportKind::Simple(name) => {
                if name == self.source.path.last() {
                    write!(f, "use {};", self.source.print(cx))
                } else {
                    write!(f, "use {} as {};", self.source.print(cx), name)
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
    pub(crate) fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| match self.did {
            Some(did) => resolved_path(f, did, &self.path, true, false, cx),
            _ => {
                for seg in &self.path.segments[..self.path.segments.len() - 1] {
                    write!(f, "{}::", seg.name)?;
                }
                let name = self.path.last();
                if let hir::def::Res::PrimTy(p) = self.path.res {
                    primitive_link(f, PrimitiveType::from(p), name.as_str(), cx)?;
                } else {
                    write!(f, "{}", name)?;
                }
                Ok(())
            }
        })
    }
}

impl clean::TypeBinding {
    pub(crate) fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| {
            f.write_str(self.assoc.name.as_str())?;
            if f.alternate() {
                write!(f, "{:#}", self.assoc.args.print(cx))?;
            } else {
                write!(f, "{}", self.assoc.args.print(cx))?;
            }
            match self.kind {
                clean::TypeBindingKind::Equality { ref term } => {
                    if f.alternate() {
                        write!(f, " = {:#}", term.print(cx))?;
                    } else {
                        write!(f, " = {}", term.print(cx))?;
                    }
                }
                clean::TypeBindingKind::Constraint { ref bounds } => {
                    if !bounds.is_empty() {
                        if f.alternate() {
                            write!(f, ": {:#}", print_generic_bounds(bounds, cx))?;
                        } else {
                            write!(f, ": {}", print_generic_bounds(bounds, cx))?;
                        }
                    }
                }
            }
            Ok(())
        })
    }
}

pub(crate) fn print_abi_with_space(abi: Abi) -> impl fmt::Display {
    display_fn(move |f| {
        let quot = if f.alternate() { "\"" } else { "&quot;" };
        match abi {
            Abi::Rust => Ok(()),
            abi => write!(f, "extern {0}{1}{0} ", quot, abi.name()),
        }
    })
}

pub(crate) fn print_default_space<'a>(v: bool) -> &'a str {
    if v { "default " } else { "" }
}

impl clean::GenericArg {
    pub(crate) fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| match self {
            clean::GenericArg::Lifetime(lt) => fmt::Display::fmt(&lt.print(), f),
            clean::GenericArg::Type(ty) => fmt::Display::fmt(&ty.print(cx), f),
            clean::GenericArg::Const(ct) => fmt::Display::fmt(&ct.print(cx.tcx()), f),
            clean::GenericArg::Infer => fmt::Display::fmt("_", f),
        })
    }
}

impl clean::types::Term {
    pub(crate) fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| match self {
            clean::types::Term::Type(ty) => fmt::Display::fmt(&ty.print(cx), f),
            clean::types::Term::Constant(ct) => fmt::Display::fmt(&ct.print(cx.tcx()), f),
        })
    }
}

pub(crate) fn display_fn(
    f: impl FnOnce(&mut fmt::Formatter<'_>) -> fmt::Result,
) -> impl fmt::Display {
    struct WithFormatter<F>(Cell<Option<F>>);

    impl<F> fmt::Display for WithFormatter<F>
    where
        F: FnOnce(&mut fmt::Formatter<'_>) -> fmt::Result,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            (self.0.take()).unwrap()(f)
        }
    }

    WithFormatter(Cell::new(Some(f)))
}
