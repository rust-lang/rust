//! HTML formatting module
//!
//! This module contains a large number of `fmt::Display` implementations for
//! various types in `rustdoc::clean`. These implementations all currently
//! assume that HTML output is desired, although it may be possible to redesign
//! them in the future to instead emit any format desired.

use std::cell::Cell;
use std::fmt;
use std::iter;

use rustc_attr::{ConstStability, StabilityLevel};
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::ty;
use rustc_middle::ty::DefIdTree;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::CRATE_DEF_INDEX;
use rustc_target::spec::abi::Abi;

use crate::clean::{
    self, types::ExternalLocation, utils::find_nearest_parent_module, ExternalCrate, ItemId,
    PrimitiveType,
};
use crate::formats::item_type::ItemType;
use crate::html::escape::Escape;
use crate::html::render::Context;

use super::url_parts_builder::UrlPartsBuilder;

crate trait Print {
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
crate struct Buffer {
    for_html: bool,
    buffer: String,
}

impl Buffer {
    crate fn empty_from(v: &Buffer) -> Buffer {
        Buffer { for_html: v.for_html, buffer: String::new() }
    }

    crate fn html() -> Buffer {
        Buffer { for_html: true, buffer: String::new() }
    }

    crate fn new() -> Buffer {
        Buffer { for_html: false, buffer: String::new() }
    }

    crate fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    crate fn into_inner(self) -> String {
        self.buffer
    }

    crate fn insert_str(&mut self, idx: usize, s: &str) {
        self.buffer.insert_str(idx, s);
    }

    crate fn push_str(&mut self, s: &str) {
        self.buffer.push_str(s);
    }

    crate fn push_buffer(&mut self, other: Buffer) {
        self.buffer.push_str(&other.buffer);
    }

    // Intended for consumption by write! and writeln! (std::fmt) but without
    // the fmt::Result return type imposed by fmt::Write (and avoiding the trait
    // import).
    crate fn write_str(&mut self, s: &str) {
        self.buffer.push_str(s);
    }

    // Intended for consumption by write! and writeln! (std::fmt) but without
    // the fmt::Result return type imposed by fmt::Write (and avoiding the trait
    // import).
    crate fn write_fmt(&mut self, v: fmt::Arguments<'_>) {
        use fmt::Write;
        self.buffer.write_fmt(v).unwrap();
    }

    crate fn to_display<T: Print>(mut self, t: T) -> String {
        t.print(&mut self);
        self.into_inner()
    }

    crate fn is_for_html(&self) -> bool {
        self.for_html
    }

    crate fn reserve(&mut self, additional: usize) {
        self.buffer.reserve(additional)
    }
}

fn comma_sep<T: fmt::Display>(items: impl Iterator<Item = T>) -> impl fmt::Display {
    display_fn(move |f| {
        for (i, item) in items.enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            fmt::Display::fmt(&item, f)?;
        }
        Ok(())
    })
}

crate fn print_generic_bounds<'a, 'tcx: 'a>(
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
    crate fn print<'a, 'tcx: 'a>(
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
                        write!(f, ":&nbsp;{}", print_generic_bounds(bounds, cx))?;
                    }
                }

                if let Some(ref ty) = default {
                    if f.alternate() {
                        write!(f, " = {:#}", ty.print(cx))?;
                    } else {
                        write!(f, "&nbsp;=&nbsp;{}", ty.print(cx))?;
                    }
                }

                Ok(())
            }
            clean::GenericParamDefKind::Const { ty, default, .. } => {
                if f.alternate() {
                    write!(f, "const {}: {:#}", self.name, ty.print(cx))?;
                } else {
                    write!(f, "const {}:&nbsp;{}", self.name, ty.print(cx))?;
                }

                if let Some(default) = default {
                    if f.alternate() {
                        write!(f, " = {:#}", default)?;
                    } else {
                        write!(f, "&nbsp;=&nbsp;{}", default)?;
                    }
                }

                Ok(())
            }
        })
    }
}

impl clean::Generics {
    crate fn print<'a, 'tcx: 'a>(
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
                write!(f, "<{:#}>", comma_sep(real_params.map(|g| g.print(cx))))
            } else {
                write!(f, "&lt;{}&gt;", comma_sep(real_params.map(|g| g.print(cx))))
            }
        })
    }
}

/// * The Generics from which to emit a where-clause.
/// * The number of spaces to indent each line with.
/// * Whether the where-clause needs to add a comma and newline after the last bound.
crate fn print_where_clause<'a, 'tcx: 'a>(
    gens: &'a clean::Generics,
    cx: &'a Context<'tcx>,
    indent: usize,
    end_newline: bool,
) -> impl fmt::Display + 'a + Captures<'tcx> {
    display_fn(move |f| {
        if gens.where_predicates.is_empty() {
            return Ok(());
        }
        let mut clause = String::new();
        if f.alternate() {
            clause.push_str(" where");
        } else {
            if end_newline {
                clause.push_str(" <span class=\"where fmt-newline\">where");
            } else {
                clause.push_str(" <span class=\"where\">where");
            }
        }
        for (i, pred) in gens.where_predicates.iter().enumerate() {
            if f.alternate() {
                clause.push(' ');
            } else {
                clause.push_str("<br>");
            }

            match pred {
                clean::WherePredicate::BoundPredicate { ty, bounds, bound_params } => {
                    let bounds = bounds;
                    let for_prefix = match bound_params.len() {
                        0 => String::new(),
                        _ if f.alternate() => {
                            format!(
                                "for&lt;{:#}&gt; ",
                                comma_sep(bound_params.iter().map(|lt| lt.print()))
                            )
                        }
                        _ => format!(
                            "for&lt;{}&gt; ",
                            comma_sep(bound_params.iter().map(|lt| lt.print()))
                        ),
                    };

                    if f.alternate() {
                        clause.push_str(&format!(
                            "{}{:#}: {:#}",
                            for_prefix,
                            ty.print(cx),
                            print_generic_bounds(bounds, cx)
                        ));
                    } else {
                        clause.push_str(&format!(
                            "{}{}: {}",
                            for_prefix,
                            ty.print(cx),
                            print_generic_bounds(bounds, cx)
                        ));
                    }
                }
                clean::WherePredicate::RegionPredicate { lifetime, bounds } => {
                    clause.push_str(&format!(
                        "{}: {}",
                        lifetime.print(),
                        bounds
                            .iter()
                            .map(|b| b.print(cx).to_string())
                            .collect::<Vec<_>>()
                            .join(" + ")
                    ));
                }
                clean::WherePredicate::EqPredicate { lhs, rhs } => {
                    if f.alternate() {
                        clause.push_str(&format!("{:#} == {:#}", lhs.print(cx), rhs.print(cx),));
                    } else {
                        clause.push_str(&format!("{} == {}", lhs.print(cx), rhs.print(cx),));
                    }
                }
            }

            if i < gens.where_predicates.len() - 1 || end_newline {
                clause.push(',');
            }
        }

        if end_newline {
            // add a space so stripping <br> tags and breaking spaces still renders properly
            if f.alternate() {
                clause.push(' ');
            } else {
                clause.push_str("&nbsp;");
            }
        }

        if !f.alternate() {
            clause.push_str("</span>");
            let padding = "&nbsp;".repeat(indent + 4);
            clause = clause.replace("<br>", &format!("<br>{}", padding));
            clause.insert_str(0, &"&nbsp;".repeat(indent.saturating_sub(1)));
            if !end_newline {
                clause.insert_str(0, "<br>");
            }
        }
        write!(f, "{}", clause)
    })
}

impl clean::Lifetime {
    crate fn print(&self) -> impl fmt::Display + '_ {
        self.0.as_str()
    }
}

impl clean::Constant {
    crate fn print(&self, tcx: TyCtxt<'_>) -> impl fmt::Display + '_ {
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
                        comma_sep(self.generic_params.iter().map(|g| g.print(cx)))
                    )?;
                } else {
                    write!(
                        f,
                        "for&lt;{}&gt; ",
                        comma_sep(self.generic_params.iter().map(|g| g.print(cx)))
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
    crate fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| match self {
            clean::GenericBound::Outlives(lt) => write!(f, "{}", lt.print()),
            clean::GenericBound::TraitBound(ty, modifier) => {
                let modifier_str = match modifier {
                    hir::TraitBoundModifier::None => "",
                    hir::TraitBoundModifier::Maybe => "?",
                    hir::TraitBoundModifier::MaybeConst => "~const",
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
                        for arg in args {
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
                        for binding in bindings {
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
                    for ty in inputs {
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
crate enum HrefError {
    /// This item is known to rustdoc, but from a crate that does not have documentation generated.
    ///
    /// This can only happen for non-local items.
    DocumentationNotBuilt,
    /// This can only happen for non-local items when `--document-private-items` is not passed.
    Private,
    // Not in external cache, href link should be in same page
    NotInExternalCache,
}

crate fn href_with_root_path(
    did: DefId,
    cx: &Context<'_>,
    root_path: Option<&str>,
) -> Result<(String, ItemType, Vec<String>), HrefError> {
    let tcx = cx.tcx();
    let def_kind = tcx.def_kind(did);
    let did = match def_kind {
        DefKind::AssocTy | DefKind::AssocFn | DefKind::AssocConst | DefKind::Variant => {
            // documented on their parent's page
            tcx.parent(did).unwrap()
        }
        _ => did,
    };
    let cache = cx.cache();
    let relative_to = &cx.current;
    fn to_module_fqp(shortty: ItemType, fqp: &[String]) -> &[String] {
        if shortty == ItemType::Module { fqp } else { &fqp[..fqp.len() - 1] }
    }

    if !did.is_local()
        && !cache.access_levels.is_public(did)
        && !cache.document_private
        && !cache.primitive_locations.values().any(|&id| id == did)
    {
        return Err(HrefError::Private);
    }

    let mut is_remote = false;
    let (fqp, shortty, mut url_parts) = match cache.paths.get(&did) {
        Some(&(ref fqp, shortty)) => (fqp, shortty, {
            let module_fqp = to_module_fqp(shortty, fqp);
            debug!(?fqp, ?shortty, ?module_fqp);
            href_relative_parts(module_fqp, relative_to)
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
                            builder.extend(module_fqp.iter().map(String::as_str));
                            builder
                        }
                        ExternalLocation::Local => href_relative_parts(module_fqp, relative_to),
                        ExternalLocation::Unknown => return Err(HrefError::DocumentationNotBuilt),
                    },
                )
            } else {
                return Err(HrefError::NotInExternalCache);
            }
        }
    };
    if !is_remote {
        if let Some(root_path) = root_path {
            let root = root_path.trim_end_matches('/');
            url_parts.push_front(root);
        }
    }
    debug!(?url_parts);
    let last = &fqp.last().unwrap()[..];
    match shortty {
        ItemType::Module => {
            url_parts.push("index.html");
        }
        _ => {
            let filename = format!("{}.{}.html", shortty.as_str(), last);
            url_parts.push(&filename);
        }
    }
    Ok((url_parts.finish(), shortty, fqp.to_vec()))
}

crate fn href(did: DefId, cx: &Context<'_>) -> Result<(String, ItemType, Vec<String>), HrefError> {
    href_with_root_path(did, cx, None)
}

/// Both paths should only be modules.
/// This is because modules get their own directories; that is, `std::vec` and `std::vec::Vec` will
/// both need `../iter/trait.Iterator.html` to get at the iterator trait.
crate fn href_relative_parts(fqp: &[String], relative_to_fqp: &[String]) -> UrlPartsBuilder {
    for (i, (f, r)) in fqp.iter().zip(relative_to_fqp.iter()).enumerate() {
        // e.g. linking to std::iter from std::vec (`dissimilar_part_count` will be 1)
        if f != r {
            let dissimilar_part_count = relative_to_fqp.len() - i;
            let fqp_module = fqp[i..fqp.len()].iter().map(String::as_str);
            return iter::repeat("..").take(dissimilar_part_count).chain(fqp_module).collect();
        }
    }
    // e.g. linking to std::sync::atomic from std::sync
    if relative_to_fqp.len() < fqp.len() {
        fqp[relative_to_fqp.len()..fqp.len()].iter().map(String::as_str).collect()
    // e.g. linking to std::sync from std::sync::atomic
    } else if fqp.len() < relative_to_fqp.len() {
        let dissimilar_part_count = relative_to_fqp.len() - fqp.len();
        iter::repeat("..").take(dissimilar_part_count).collect()
    // linking to the same module
    } else {
        UrlPartsBuilder::new()
    }
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
            write!(w, "{}::", seg.name)?;
        }
    }
    if w.alternate() {
        write!(w, "{}{:#}", &last.name, last.args.print(cx))?;
    } else {
        let path = if use_absolute {
            if let Ok((_, _, fqp)) = href(did, cx) {
                format!(
                    "{}::{}",
                    fqp[..fqp.len() - 1].join("::"),
                    anchor(did, fqp.last().unwrap(), cx)
                )
            } else {
                last.name.to_string()
            }
        } else {
            anchor(did, last.name.as_str(), cx).to_string()
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
    let m = &cx.cache();
    let mut needs_termination = false;
    if !f.alternate() {
        match m.primitive_locations.get(&prim) {
            Some(&def_id) if def_id.is_local() => {
                let len = cx.current.len();
                let len = if len == 0 { 0 } else { len - 1 };
                write!(
                    f,
                    "<a class=\"primitive\" href=\"{}primitive.{}.html\">",
                    "../".repeat(len),
                    prim.as_sym()
                )?;
                needs_termination = true;
            }
            Some(&def_id) => {
                let cname_sym;
                let loc = match m.extern_locations[&def_id.krate] {
                    ExternalLocation::Remote(ref s) => {
                        cname_sym = ExternalCrate { crate_num: def_id.krate }.name(cx.tcx());
                        Some(vec![s.trim_end_matches('/'), cname_sym.as_str()])
                    }
                    ExternalLocation::Local => {
                        cname_sym = ExternalCrate { crate_num: def_id.krate }.name(cx.tcx());
                        Some(if cx.current.first().map(|x| &x[..]) == Some(cname_sym.as_str()) {
                            iter::repeat("..").take(cx.current.len() - 1).collect()
                        } else {
                            let cname = iter::once(cname_sym.as_str());
                            iter::repeat("..").take(cx.current.len()).chain(cname).collect()
                        })
                    }
                    ExternalLocation::Unknown => None,
                };
                if let Some(loc) = loc {
                    write!(
                        f,
                        "<a class=\"primitive\" href=\"{}/primitive.{}.html\">",
                        loc.join("/"),
                        prim.as_sym()
                    )?;
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

crate fn anchor<'a, 'cx: 'a>(
    did: DefId,
    text: &'a str,
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
                fqp.join("::"),
                text
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
                &[ref one] => {
                    primitive_link(f, PrimitiveType::Tuple, "(", cx)?;
                    // Carry `f.alternate()` into this display w/o branching manually.
                    fmt::Display::fmt(&one.print(cx), f)?;
                    primitive_link(f, PrimitiveType::Tuple, ",)", cx)
                }
                many => {
                    primitive_link(f, PrimitiveType::Tuple, "(", cx)?;
                    for (i, item) in many.iter().enumerate() {
                        if i != 0 {
                            write!(f, ", ")?;
                        }
                        fmt::Display::fmt(&item.print(cx), f)?;
                    }
                    primitive_link(f, PrimitiveType::Tuple, ")", cx)
                }
            }
        }
        clean::Slice(ref t) => {
            primitive_link(f, PrimitiveType::Slice, "[", cx)?;
            fmt::Display::fmt(&t.print(cx), f)?;
            primitive_link(f, PrimitiveType::Slice, "]", cx)
        }
        clean::Array(ref t, ref n) => {
            primitive_link(f, PrimitiveType::Array, "[", cx)?;
            fmt::Display::fmt(&t.print(cx), f)?;
            if f.alternate() {
                primitive_link(f, PrimitiveType::Array, &format!("; {}]", n), cx)
            } else {
                primitive_link(f, PrimitiveType::Array, &format!("; {}]", Escape(n)), cx)
            }
        }
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
            let amp = if f.alternate() { "&".to_string() } else { "&amp;".to_string() };
            match **ty {
                clean::Slice(ref bt) => {
                    // `BorrowedRef{ ... Slice(T) }` is `&[T]`
                    match **bt {
                        clean::Generic(_) => {
                            if f.alternate() {
                                primitive_link(
                                    f,
                                    PrimitiveType::Slice,
                                    &format!("{}{}{}[{:#}]", amp, lt, m, bt.print(cx)),
                                    cx,
                                )
                            } else {
                                primitive_link(
                                    f,
                                    PrimitiveType::Slice,
                                    &format!("{}{}{}[{}]", amp, lt, m, bt.print(cx)),
                                    cx,
                                )
                            }
                        }
                        _ => {
                            primitive_link(
                                f,
                                PrimitiveType::Slice,
                                &format!("{}{}{}[", amp, lt, m),
                                cx,
                            )?;
                            if f.alternate() {
                                write!(f, "{:#}", bt.print(cx))?;
                            } else {
                                write!(f, "{}", bt.print(cx))?;
                            }
                            primitive_link(f, PrimitiveType::Slice, "]", cx)
                        }
                    }
                }
                clean::DynTrait(ref bounds, ref trait_lt)
                    if bounds.len() > 1 || trait_lt.is_some() =>
                {
                    write!(f, "{}{}{}(", amp, lt, m)?;
                    fmt_type(ty, f, use_absolute, cx)?;
                    write!(f, ")")
                }
                clean::Generic(..) => {
                    primitive_link(
                        f,
                        PrimitiveType::Reference,
                        &format!("{}{}{}", amp, lt, m),
                        cx,
                    )?;
                    fmt_type(ty, f, use_absolute, cx)
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
        clean::QPath { ref name, ref self_type, ref trait_, ref self_def_id } => {
            let should_show_cast = !trait_.segments.is_empty()
                && self_def_id
                    .zip(Some(trait_.def_id()))
                    .map_or(!self_type.is_self_type(), |(id, trait_)| id != trait_);
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
                                    title=\"type {path}::{name}\">{name}</a>",
                        url = url,
                        shortty = ItemType::AssocType,
                        name = name,
                        path = path.join("::")
                    )?;
                }
                _ => write!(f, "{}", name)?,
            }
            Ok(())
        }
    }
}

impl clean::Type {
    crate fn print<'b, 'a: 'b, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'b + Captures<'tcx> {
        display_fn(move |f| fmt_type(self, f, false, cx))
    }
}

impl clean::Path {
    crate fn print<'b, 'a: 'b, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'b + Captures<'tcx> {
        display_fn(move |f| resolved_path(f, self.def_id(), self, false, false, cx))
    }
}

impl clean::Impl {
    crate fn print<'a, 'tcx: 'a>(
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

            if let Some(ref ty) = self.kind.as_blanket_ty() {
                fmt_type(ty, f, use_absolute, cx)?;
            } else {
                fmt_type(&self.for_, f, use_absolute, cx)?;
            }

            fmt::Display::fmt(&print_where_clause(&self.generics, cx, 0, true), f)?;
            Ok(())
        })
    }
}

impl clean::Arguments {
    crate fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| {
            for (i, input) in self.values.iter().enumerate() {
                if !input.name.is_empty() {
                    write!(f, "{}: ", input.name)?;
                }
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
    crate fn print<'a, 'tcx: 'a>(
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
                    comma_sep(self.generic_params.iter().map(|g| g.print(cx)))
                )
            } else {
                Ok(())
            }
        })
    }
}

impl clean::FnDecl {
    crate fn print<'b, 'a: 'b, 'tcx: 'a>(
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
    ///   <br>Used to determine line-wrapping.
    /// * `indent`: The number of spaces to indent each successive line with, if line-wrapping is
    ///   necessary.
    /// * `asyncness`: Whether the function is async or not.
    crate fn full_print<'a, 'tcx: 'a>(
        &'a self,
        header_len: usize,
        indent: usize,
        asyncness: hir::IsAsync,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| self.inner_full_print(header_len, indent, asyncness, f, cx))
    }

    fn inner_full_print(
        &self,
        header_len: usize,
        indent: usize,
        asyncness: hir::IsAsync,
        f: &mut fmt::Formatter<'_>,
        cx: &Context<'_>,
    ) -> fmt::Result {
        let amp = if f.alternate() { "&" } else { "&amp;" };
        let mut args = String::new();
        let mut args_plain = String::new();
        for (i, input) in self.inputs.values.iter().enumerate() {
            if i == 0 {
                args.push_str("<br>");
            }

            if let Some(selfty) = input.to_self() {
                match selfty {
                    clean::SelfValue => {
                        args.push_str("self");
                        args_plain.push_str("self");
                    }
                    clean::SelfBorrowed(Some(ref lt), mtbl) => {
                        args.push_str(&format!(
                            "{}{} {}self",
                            amp,
                            lt.print(),
                            mtbl.print_with_space()
                        ));
                        args_plain.push_str(&format!(
                            "&{} {}self",
                            lt.print(),
                            mtbl.print_with_space()
                        ));
                    }
                    clean::SelfBorrowed(None, mtbl) => {
                        args.push_str(&format!("{}{}self", amp, mtbl.print_with_space()));
                        args_plain.push_str(&format!("&{}self", mtbl.print_with_space()));
                    }
                    clean::SelfExplicit(ref typ) => {
                        if f.alternate() {
                            args.push_str(&format!("self: {:#}", typ.print(cx)));
                        } else {
                            args.push_str(&format!("self: {}", typ.print(cx)));
                        }
                        args_plain.push_str(&format!("self: {:#}", typ.print(cx)));
                    }
                }
            } else {
                if i > 0 {
                    args.push_str(" <br>");
                    args_plain.push(' ');
                }
                if input.is_const {
                    args.push_str("const ");
                    args_plain.push_str("const ");
                }
                if !input.name.is_empty() {
                    args.push_str(&format!("{}: ", input.name));
                    args_plain.push_str(&format!("{}: ", input.name));
                }

                if f.alternate() {
                    args.push_str(&format!("{:#}", input.type_.print(cx)));
                } else {
                    args.push_str(&input.type_.print(cx).to_string());
                }
                args_plain.push_str(&format!("{:#}", input.type_.print(cx)));
            }
            if i + 1 < self.inputs.values.len() {
                args.push(',');
                args_plain.push(',');
            }
        }

        let mut args_plain = format!("({})", args_plain);

        if self.c_variadic {
            args.push_str(",<br> ...");
            args_plain.push_str(", ...");
        }

        let arrow_plain;
        let arrow = if let hir::IsAsync::Async = asyncness {
            let output = self.sugared_async_return_type();
            arrow_plain = format!("{:#}", output.print(cx));
            if f.alternate() { arrow_plain.clone() } else { format!("{}", output.print(cx)) }
        } else {
            arrow_plain = format!("{:#}", self.output.print(cx));
            if f.alternate() { arrow_plain.clone() } else { format!("{}", self.output.print(cx)) }
        };

        let declaration_len = header_len + args_plain.len() + arrow_plain.len();
        let output = if declaration_len > 80 {
            let full_pad = format!("<br>{}", "&nbsp;".repeat(indent + 4));
            let close_pad = format!("<br>{}", "&nbsp;".repeat(indent));
            format!(
                "({args}{close}){arrow}",
                args = args.replace("<br>", &full_pad),
                close = close_pad,
                arrow = arrow
            )
        } else {
            format!("({args}){arrow}", args = args.replace("<br>", ""), arrow = arrow)
        };

        if f.alternate() {
            write!(f, "{}", output.replace("<br>", "\n"))
        } else {
            write!(f, "{}", output)
        }
    }
}

impl clean::Visibility {
    crate fn print_with_space<'a, 'tcx: 'a>(
        self,
        item_did: ItemId,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        let to_print = match self {
            clean::Public => "pub ".to_owned(),
            clean::Inherited => String::new(),
            clean::Visibility::Restricted(vis_did) => {
                // FIXME(camelid): This may not work correctly if `item_did` is a module.
                //                 However, rustdoc currently never displays a module's
                //                 visibility, so it shouldn't matter.
                let parent_module = find_nearest_parent_module(cx.tcx(), item_did.expect_def_id());

                if vis_did.index == CRATE_DEF_INDEX {
                    "pub(crate) ".to_owned()
                } else if parent_module == Some(vis_did) {
                    // `pub(in foo)` where `foo` is the parent module
                    // is the same as no visibility modifier
                    String::new()
                } else if parent_module
                    .map(|parent| find_nearest_parent_module(cx.tcx(), parent))
                    .flatten()
                    == Some(vis_did)
                {
                    "pub(super) ".to_owned()
                } else {
                    let path = cx.tcx().def_path(vis_did);
                    debug!("path={:?}", path);
                    // modified from `resolved_path()` to work with `DefPathData`
                    let last_name = path.data.last().unwrap().data.get_opt_name().unwrap();
                    let anchor = anchor(vis_did, last_name.as_str(), cx).to_string();

                    let mut s = "pub(in ".to_owned();
                    for seg in &path.data[..path.data.len() - 1] {
                        s.push_str(&format!("{}::", seg.data.get_opt_name().unwrap()));
                    }
                    s.push_str(&format!("{}) ", anchor));
                    s
                }
            }
        };
        display_fn(move |f| f.write_str(&to_print))
    }

    /// This function is the same as print_with_space, except that it renders no links.
    /// It's used for macros' rendered source view, which is syntax highlighted and cannot have
    /// any HTML in it.
    crate fn to_src_with_space<'a, 'tcx: 'a>(
        self,
        tcx: TyCtxt<'tcx>,
        item_did: DefId,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        let to_print = match self {
            clean::Public => "pub ".to_owned(),
            clean::Inherited => String::new(),
            clean::Visibility::Restricted(vis_did) => {
                // FIXME(camelid): This may not work correctly if `item_did` is a module.
                //                 However, rustdoc currently never displays a module's
                //                 visibility, so it shouldn't matter.
                let parent_module = find_nearest_parent_module(tcx, item_did);

                if vis_did.index == CRATE_DEF_INDEX {
                    "pub(crate) ".to_owned()
                } else if parent_module == Some(vis_did) {
                    // `pub(in foo)` where `foo` is the parent module
                    // is the same as no visibility modifier
                    String::new()
                } else if parent_module
                    .map(|parent| find_nearest_parent_module(tcx, parent))
                    .flatten()
                    == Some(vis_did)
                {
                    "pub(super) ".to_owned()
                } else {
                    format!("pub(in {}) ", tcx.def_path_str(vis_did))
                }
            }
        };
        display_fn(move |f| f.write_str(&to_print))
    }
}

crate trait PrintWithSpace {
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

crate fn print_constness_with_space(c: &hir::Constness, s: Option<ConstStability>) -> &'static str {
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
    crate fn print<'a, 'tcx: 'a>(
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
    crate fn print<'a, 'tcx: 'a>(
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
    crate fn print<'a, 'tcx: 'a>(
        &'a self,
        cx: &'a Context<'tcx>,
    ) -> impl fmt::Display + 'a + Captures<'tcx> {
        display_fn(move |f| {
            f.write_str(self.name.as_str())?;
            match self.kind {
                clean::TypeBindingKind::Equality { ref ty } => {
                    if f.alternate() {
                        write!(f, " = {:#}", ty.print(cx))?;
                    } else {
                        write!(f, " = {}", ty.print(cx))?;
                    }
                }
                clean::TypeBindingKind::Constraint { ref bounds } => {
                    if !bounds.is_empty() {
                        if f.alternate() {
                            write!(f, ": {:#}", print_generic_bounds(bounds, cx))?;
                        } else {
                            write!(f, ":&nbsp;{}", print_generic_bounds(bounds, cx))?;
                        }
                    }
                }
            }
            Ok(())
        })
    }
}

crate fn print_abi_with_space(abi: Abi) -> impl fmt::Display {
    display_fn(move |f| {
        let quot = if f.alternate() { "\"" } else { "&quot;" };
        match abi {
            Abi::Rust => Ok(()),
            abi => write!(f, "extern {0}{1}{0} ", quot, abi.name()),
        }
    })
}

crate fn print_default_space<'a>(v: bool) -> &'a str {
    if v { "default " } else { "" }
}

impl clean::GenericArg {
    crate fn print<'a, 'tcx: 'a>(
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

crate fn display_fn(f: impl FnOnce(&mut fmt::Formatter<'_>) -> fmt::Result) -> impl fmt::Display {
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
