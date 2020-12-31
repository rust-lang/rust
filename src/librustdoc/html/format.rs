//! HTML formatting module
//!
//! This module contains a large number of `fmt::Display` implementations for
//! various types in `rustdoc::clean`. These implementations all currently
//! assume that HTML output is desired, although it may be possible to redesign
//! them in the future to instead emit any format desired.

use std::borrow::Cow;
use std::cell::Cell;
use std::fmt;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::{DefId, CRATE_DEF_INDEX};
use rustc_target::spec::abi::Abi;

use crate::clean::{self, utils::find_nearest_parent_module, PrimitiveType};
use crate::formats::cache::cache;
use crate::formats::item_type::ItemType;
use crate::html::escape::Escape;
use crate::html::render::cache::ExternalLocation;
use crate::html::render::CURRENT_DEPTH;

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

    crate fn from_display<T: std::fmt::Display>(&mut self, t: T) {
        if self.for_html {
            write!(self, "{}", t);
        } else {
            write!(self, "{:#}", t);
        }
    }

    crate fn is_for_html(&self) -> bool {
        self.for_html
    }
}

/// Wrapper struct for properly emitting a function or method declaration.
crate struct Function<'a> {
    /// The declaration to emit.
    crate decl: &'a clean::FnDecl,
    /// The length of the function header and name. In other words, the number of characters in the
    /// function declaration up to but not including the parentheses.
    ///
    /// Used to determine line-wrapping.
    crate header_len: usize,
    /// The number of spaces to indent each successive line with, if line-wrapping is necessary.
    crate indent: usize,
    /// Whether the function is async or not.
    crate asyncness: hir::IsAsync,
}

/// Wrapper struct for emitting a where-clause from Generics.
crate struct WhereClause<'a> {
    /// The Generics from which to emit a where-clause.
    crate gens: &'a clean::Generics,
    /// The number of spaces to indent each line with.
    crate indent: usize,
    /// Whether the where-clause needs to add a comma and newline after the last bound.
    crate end_newline: bool,
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

crate fn print_generic_bounds(bounds: &[clean::GenericBound]) -> impl fmt::Display + '_ {
    display_fn(move |f| {
        let mut bounds_dup = FxHashSet::default();

        for (i, bound) in
            bounds.iter().filter(|b| bounds_dup.insert(b.print().to_string())).enumerate()
        {
            if i > 0 {
                f.write_str(" + ")?;
            }
            fmt::Display::fmt(&bound.print(), f)?;
        }
        Ok(())
    })
}

impl clean::GenericParamDef {
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| match self.kind {
            clean::GenericParamDefKind::Lifetime => write!(f, "{}", self.name),
            clean::GenericParamDefKind::Type { ref bounds, ref default, .. } => {
                f.write_str(&*self.name.as_str())?;

                if !bounds.is_empty() {
                    if f.alternate() {
                        write!(f, ": {:#}", print_generic_bounds(bounds))?;
                    } else {
                        write!(f, ":&nbsp;{}", print_generic_bounds(bounds))?;
                    }
                }

                if let Some(ref ty) = default {
                    if f.alternate() {
                        write!(f, " = {:#}", ty.print())?;
                    } else {
                        write!(f, "&nbsp;=&nbsp;{}", ty.print())?;
                    }
                }

                Ok(())
            }
            clean::GenericParamDefKind::Const { ref ty, .. } => {
                if f.alternate() {
                    write!(f, "const {}: {:#}", self.name, ty.print())
                } else {
                    write!(f, "const {}:&nbsp;{}", self.name, ty.print())
                }
            }
        })
    }
}

impl clean::Generics {
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| {
            let real_params =
                self.params.iter().filter(|p| !p.is_synthetic_type_param()).collect::<Vec<_>>();
            if real_params.is_empty() {
                return Ok(());
            }
            if f.alternate() {
                write!(f, "<{:#}>", comma_sep(real_params.iter().map(|g| g.print())))
            } else {
                write!(f, "&lt;{}&gt;", comma_sep(real_params.iter().map(|g| g.print())))
            }
        })
    }
}

impl<'a> fmt::Display for WhereClause<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &WhereClause { gens, indent, end_newline } = self;
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
                clean::WherePredicate::BoundPredicate { ty, bounds } => {
                    let bounds = bounds;
                    if f.alternate() {
                        clause.push_str(&format!(
                            "{:#}: {:#}",
                            ty.print(),
                            print_generic_bounds(bounds)
                        ));
                    } else {
                        clause.push_str(&format!(
                            "{}: {}",
                            ty.print(),
                            print_generic_bounds(bounds)
                        ));
                    }
                }
                clean::WherePredicate::RegionPredicate { lifetime, bounds } => {
                    clause.push_str(&format!(
                        "{}: {}",
                        lifetime.print(),
                        bounds
                            .iter()
                            .map(|b| b.print().to_string())
                            .collect::<Vec<_>>()
                            .join(" + ")
                    ));
                }
                clean::WherePredicate::EqPredicate { lhs, rhs } => {
                    if f.alternate() {
                        clause.push_str(&format!("{:#} == {:#}", lhs.print(), rhs.print()));
                    } else {
                        clause.push_str(&format!("{} == {}", lhs.print(), rhs.print()));
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
    }
}

impl clean::Lifetime {
    crate fn print(&self) -> impl fmt::Display + '_ {
        self.get_ref()
    }
}

impl clean::Constant {
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| {
            if f.alternate() {
                f.write_str(&self.expr)
            } else {
                write!(f, "{}", Escape(&self.expr))
            }
        })
    }
}

impl clean::PolyTrait {
    fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| {
            if !self.generic_params.is_empty() {
                if f.alternate() {
                    write!(
                        f,
                        "for<{:#}> ",
                        comma_sep(self.generic_params.iter().map(|g| g.print()))
                    )?;
                } else {
                    write!(
                        f,
                        "for&lt;{}&gt; ",
                        comma_sep(self.generic_params.iter().map(|g| g.print()))
                    )?;
                }
            }
            if f.alternate() {
                write!(f, "{:#}", self.trait_.print())
            } else {
                write!(f, "{}", self.trait_.print())
            }
        })
    }
}

impl clean::GenericBound {
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| match self {
            clean::GenericBound::Outlives(lt) => write!(f, "{}", lt.print()),
            clean::GenericBound::TraitBound(ty, modifier) => {
                let modifier_str = match modifier {
                    hir::TraitBoundModifier::None => "",
                    hir::TraitBoundModifier::Maybe => "?",
                    hir::TraitBoundModifier::MaybeConst => "?const",
                };
                if f.alternate() {
                    write!(f, "{}{:#}", modifier_str, ty.print())
                } else {
                    write!(f, "{}{}", modifier_str, ty.print())
                }
            }
        })
    }
}

impl clean::GenericArgs {
    fn print(&self) -> impl fmt::Display + '_ {
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
                                write!(f, "{:#}", arg.print())?;
                            } else {
                                write!(f, "{}", arg.print())?;
                            }
                        }
                        for binding in bindings {
                            if comma {
                                f.write_str(", ")?;
                            }
                            comma = true;
                            if f.alternate() {
                                write!(f, "{:#}", binding.print())?;
                            } else {
                                write!(f, "{}", binding.print())?;
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
                            write!(f, "{:#}", ty.print())?;
                        } else {
                            write!(f, "{}", ty.print())?;
                        }
                    }
                    f.write_str(")")?;
                    if let Some(ref ty) = *output {
                        if f.alternate() {
                            write!(f, " -> {:#}", ty.print())?;
                        } else {
                            write!(f, " -&gt; {}", ty.print())?;
                        }
                    }
                }
            }
            Ok(())
        })
    }
}

impl clean::PathSegment {
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| {
            if f.alternate() {
                write!(f, "{}{:#}", self.name, self.args.print())
            } else {
                write!(f, "{}{}", self.name, self.args.print())
            }
        })
    }
}

impl clean::Path {
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| {
            if self.global {
                f.write_str("::")?
            }

            for (i, seg) in self.segments.iter().enumerate() {
                if i > 0 {
                    f.write_str("::")?
                }
                if f.alternate() {
                    write!(f, "{:#}", seg.print())?;
                } else {
                    write!(f, "{}", seg.print())?;
                }
            }
            Ok(())
        })
    }
}

crate fn href(did: DefId) -> Option<(String, ItemType, Vec<String>)> {
    let cache = cache();
    if !did.is_local() && !cache.access_levels.is_public(did) && !cache.document_private {
        return None;
    }

    let depth = CURRENT_DEPTH.with(|l| l.get());
    let (fqp, shortty, mut url) = match cache.paths.get(&did) {
        Some(&(ref fqp, shortty)) => (fqp, shortty, "../".repeat(depth)),
        None => {
            let &(ref fqp, shortty) = cache.external_paths.get(&did)?;
            (
                fqp,
                shortty,
                match cache.extern_locations[&did.krate] {
                    (.., ExternalLocation::Remote(ref s)) => s.to_string(),
                    (.., ExternalLocation::Local) => "../".repeat(depth),
                    (.., ExternalLocation::Unknown) => return None,
                },
            )
        }
    };
    for component in &fqp[..fqp.len() - 1] {
        url.push_str(component);
        url.push('/');
    }
    match shortty {
        ItemType::Module => {
            url.push_str(fqp.last().unwrap());
            url.push_str("/index.html");
        }
        _ => {
            url.push_str(shortty.as_str());
            url.push('.');
            url.push_str(fqp.last().unwrap());
            url.push_str(".html");
        }
    }
    Some((url, shortty, fqp.to_vec()))
}

/// Used when rendering a `ResolvedPath` structure. This invokes the `path`
/// rendering function with the necessary arguments for linking to a local path.
fn resolved_path(
    w: &mut fmt::Formatter<'_>,
    did: DefId,
    path: &clean::Path,
    print_all: bool,
    use_absolute: bool,
) -> fmt::Result {
    let last = path.segments.last().unwrap();

    if print_all {
        for seg in &path.segments[..path.segments.len() - 1] {
            write!(w, "{}::", seg.name)?;
        }
    }
    if w.alternate() {
        write!(w, "{}{:#}", &last.name, last.args.print())?;
    } else {
        let path = if use_absolute {
            if let Some((_, _, fqp)) = href(did) {
                format!("{}::{}", fqp[..fqp.len() - 1].join("::"), anchor(did, fqp.last().unwrap()))
            } else {
                last.name.to_string()
            }
        } else {
            anchor(did, &*last.name.as_str()).to_string()
        };
        write!(w, "{}{}", path, last.args.print())?;
    }
    Ok(())
}

fn primitive_link(
    f: &mut fmt::Formatter<'_>,
    prim: clean::PrimitiveType,
    name: &str,
) -> fmt::Result {
    let m = cache();
    let mut needs_termination = false;
    if !f.alternate() {
        match m.primitive_locations.get(&prim) {
            Some(&def_id) if def_id.is_local() => {
                let len = CURRENT_DEPTH.with(|s| s.get());
                let len = if len == 0 { 0 } else { len - 1 };
                write!(
                    f,
                    "<a class=\"primitive\" href=\"{}primitive.{}.html\">",
                    "../".repeat(len),
                    prim.to_url_str()
                )?;
                needs_termination = true;
            }
            Some(&def_id) => {
                let loc = match m.extern_locations[&def_id.krate] {
                    (ref cname, _, ExternalLocation::Remote(ref s)) => Some((cname, s.to_string())),
                    (ref cname, _, ExternalLocation::Local) => {
                        let len = CURRENT_DEPTH.with(|s| s.get());
                        Some((cname, "../".repeat(len)))
                    }
                    (.., ExternalLocation::Unknown) => None,
                };
                if let Some((cname, root)) = loc {
                    write!(
                        f,
                        "<a class=\"primitive\" href=\"{}{}/primitive.{}.html\">",
                        root,
                        cname,
                        prim.to_url_str()
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
fn tybounds(param_names: &Option<Vec<clean::GenericBound>>) -> impl fmt::Display + '_ {
    display_fn(move |f| match *param_names {
        Some(ref params) => {
            for param in params {
                write!(f, " + ")?;
                fmt::Display::fmt(&param.print(), f)?;
            }
            Ok(())
        }
        None => Ok(()),
    })
}

crate fn anchor(did: DefId, text: &str) -> impl fmt::Display + '_ {
    display_fn(move |f| {
        if let Some((url, short_ty, fqp)) = href(did) {
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

fn fmt_type(t: &clean::Type, f: &mut fmt::Formatter<'_>, use_absolute: bool) -> fmt::Result {
    match *t {
        clean::Generic(name) => write!(f, "{}", name),
        clean::ResolvedPath { did, ref param_names, ref path, is_generic } => {
            if param_names.is_some() {
                f.write_str("dyn ")?;
            }
            // Paths like `T::Output` and `Self::Output` should be rendered with all segments.
            resolved_path(f, did, path, is_generic, use_absolute)?;
            fmt::Display::fmt(&tybounds(param_names), f)
        }
        clean::Infer => write!(f, "_"),
        clean::Primitive(prim) => primitive_link(f, prim, prim.as_str()),
        clean::BareFunction(ref decl) => {
            if f.alternate() {
                write!(
                    f,
                    "{}{:#}fn{:#}{:#}",
                    decl.unsafety.print_with_space(),
                    print_abi_with_space(decl.abi),
                    decl.print_generic_params(),
                    decl.decl.print()
                )
            } else {
                write!(
                    f,
                    "{}{}",
                    decl.unsafety.print_with_space(),
                    print_abi_with_space(decl.abi)
                )?;
                primitive_link(f, PrimitiveType::Fn, "fn")?;
                write!(f, "{}{}", decl.print_generic_params(), decl.decl.print())
            }
        }
        clean::Tuple(ref typs) => {
            match &typs[..] {
                &[] => primitive_link(f, PrimitiveType::Unit, "()"),
                &[ref one] => {
                    primitive_link(f, PrimitiveType::Tuple, "(")?;
                    // Carry `f.alternate()` into this display w/o branching manually.
                    fmt::Display::fmt(&one.print(), f)?;
                    primitive_link(f, PrimitiveType::Tuple, ",)")
                }
                many => {
                    primitive_link(f, PrimitiveType::Tuple, "(")?;
                    for (i, item) in many.iter().enumerate() {
                        if i != 0 {
                            write!(f, ", ")?;
                        }
                        fmt::Display::fmt(&item.print(), f)?;
                    }
                    primitive_link(f, PrimitiveType::Tuple, ")")
                }
            }
        }
        clean::Slice(ref t) => {
            primitive_link(f, PrimitiveType::Slice, "[")?;
            fmt::Display::fmt(&t.print(), f)?;
            primitive_link(f, PrimitiveType::Slice, "]")
        }
        clean::Array(ref t, ref n) => {
            primitive_link(f, PrimitiveType::Array, "[")?;
            fmt::Display::fmt(&t.print(), f)?;
            if f.alternate() {
                primitive_link(f, PrimitiveType::Array, &format!("; {}]", n))
            } else {
                primitive_link(f, PrimitiveType::Array, &format!("; {}]", Escape(n)))
            }
        }
        clean::Never => primitive_link(f, PrimitiveType::Never, "!"),
        clean::RawPointer(m, ref t) => {
            let m = match m {
                hir::Mutability::Mut => "mut",
                hir::Mutability::Not => "const",
            };
            match **t {
                clean::Generic(_) | clean::ResolvedPath { is_generic: true, .. } => {
                    if f.alternate() {
                        primitive_link(
                            f,
                            clean::PrimitiveType::RawPointer,
                            &format!("*{} {:#}", m, t.print()),
                        )
                    } else {
                        primitive_link(
                            f,
                            clean::PrimitiveType::RawPointer,
                            &format!("*{} {}", m, t.print()),
                        )
                    }
                }
                _ => {
                    primitive_link(f, clean::PrimitiveType::RawPointer, &format!("*{} ", m))?;
                    fmt::Display::fmt(&t.print(), f)
                }
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
                                    &format!("{}{}{}[{:#}]", amp, lt, m, bt.print()),
                                )
                            } else {
                                primitive_link(
                                    f,
                                    PrimitiveType::Slice,
                                    &format!("{}{}{}[{}]", amp, lt, m, bt.print()),
                                )
                            }
                        }
                        _ => {
                            primitive_link(
                                f,
                                PrimitiveType::Slice,
                                &format!("{}{}{}[", amp, lt, m),
                            )?;
                            if f.alternate() {
                                write!(f, "{:#}", bt.print())?;
                            } else {
                                write!(f, "{}", bt.print())?;
                            }
                            primitive_link(f, PrimitiveType::Slice, "]")
                        }
                    }
                }
                clean::ResolvedPath { param_names: Some(ref v), .. } if !v.is_empty() => {
                    write!(f, "{}{}{}(", amp, lt, m)?;
                    fmt_type(&ty, f, use_absolute)?;
                    write!(f, ")")
                }
                clean::Generic(..) => {
                    primitive_link(f, PrimitiveType::Reference, &format!("{}{}{}", amp, lt, m))?;
                    fmt_type(&ty, f, use_absolute)
                }
                _ => {
                    write!(f, "{}{}{}", amp, lt, m)?;
                    fmt_type(&ty, f, use_absolute)
                }
            }
        }
        clean::ImplTrait(ref bounds) => {
            if f.alternate() {
                write!(f, "impl {:#}", print_generic_bounds(bounds))
            } else {
                write!(f, "impl {}", print_generic_bounds(bounds))
            }
        }
        clean::QPath { ref name, ref self_type, ref trait_ } => {
            let should_show_cast = match *trait_ {
                box clean::ResolvedPath { ref path, .. } => {
                    !path.segments.is_empty() && !self_type.is_self_type()
                }
                _ => true,
            };
            if f.alternate() {
                if should_show_cast {
                    write!(f, "<{:#} as {:#}>::", self_type.print(), trait_.print())?
                } else {
                    write!(f, "{:#}::", self_type.print())?
                }
            } else {
                if should_show_cast {
                    write!(f, "&lt;{} as {}&gt;::", self_type.print(), trait_.print())?
                } else {
                    write!(f, "{}::", self_type.print())?
                }
            };
            match *trait_ {
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
                box clean::ResolvedPath { did, ref param_names, .. } => {
                    match href(did) {
                        Some((ref url, _, ref path)) if !f.alternate() => {
                            write!(
                                f,
                                "<a class=\"type\" href=\"{url}#{shortty}.{name}\" \
                                    title=\"type {path}::{name}\">{name}</a>",
                                url = url,
                                shortty = ItemType::AssocType,
                                name = name,
                                path = path.join("::")
                            )?;
                        }
                        _ => write!(f, "{}", name)?,
                    }

                    // FIXME: `param_names` are not rendered, and this seems bad?
                    drop(param_names);
                    Ok(())
                }
                _ => write!(f, "{}", name),
            }
        }
    }
}

impl clean::Type {
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| fmt_type(self, f, false))
    }
}

impl clean::Impl {
    crate fn print(&self) -> impl fmt::Display + '_ {
        self.print_inner(true, false)
    }

    fn print_inner(&self, link_trait: bool, use_absolute: bool) -> impl fmt::Display + '_ {
        display_fn(move |f| {
            if f.alternate() {
                write!(f, "impl{:#} ", self.generics.print())?;
            } else {
                write!(f, "impl{} ", self.generics.print())?;
            }

            if let Some(ref ty) = self.trait_ {
                if self.polarity == Some(clean::ImplPolarity::Negative) {
                    write!(f, "!")?;
                }

                if link_trait {
                    fmt::Display::fmt(&ty.print(), f)?;
                } else {
                    match ty {
                        clean::ResolvedPath {
                            param_names: None, path, is_generic: false, ..
                        } => {
                            let last = path.segments.last().unwrap();
                            fmt::Display::fmt(&last.name, f)?;
                            fmt::Display::fmt(&last.args.print(), f)?;
                        }
                        _ => unreachable!(),
                    }
                }
                write!(f, " for ")?;
            }

            if let Some(ref ty) = self.blanket_impl {
                fmt_type(ty, f, use_absolute)?;
            } else {
                fmt_type(&self.for_, f, use_absolute)?;
            }

            fmt::Display::fmt(
                &WhereClause { gens: &self.generics, indent: 0, end_newline: true },
                f,
            )?;
            Ok(())
        })
    }
}

// The difference from above is that trait is not hyperlinked.
crate fn fmt_impl_for_trait_page(i: &clean::Impl, f: &mut Buffer, use_absolute: bool) {
    f.from_display(i.print_inner(false, use_absolute))
}

impl clean::Arguments {
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| {
            for (i, input) in self.values.iter().enumerate() {
                if !input.name.is_empty() {
                    write!(f, "{}: ", input.name)?;
                }
                if f.alternate() {
                    write!(f, "{:#}", input.type_.print())?;
                } else {
                    write!(f, "{}", input.type_.print())?;
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
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| match self {
            clean::Return(clean::Tuple(tys)) if tys.is_empty() => Ok(()),
            clean::Return(ty) if f.alternate() => write!(f, " -> {:#}", ty.print()),
            clean::Return(ty) => write!(f, " -&gt; {}", ty.print()),
            clean::DefaultReturn => Ok(()),
        })
    }
}

impl clean::BareFunctionDecl {
    fn print_generic_params(&self) -> impl fmt::Display + '_ {
        comma_sep(self.generic_params.iter().map(|g| g.print()))
    }
}

impl clean::FnDecl {
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| {
            let ellipsis = if self.c_variadic { ", ..." } else { "" };
            if f.alternate() {
                write!(
                    f,
                    "({args:#}{ellipsis}){arrow:#}",
                    args = self.inputs.print(),
                    ellipsis = ellipsis,
                    arrow = self.output.print()
                )
            } else {
                write!(
                    f,
                    "({args}{ellipsis}){arrow}",
                    args = self.inputs.print(),
                    ellipsis = ellipsis,
                    arrow = self.output.print()
                )
            }
        })
    }
}

impl Function<'_> {
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| {
            let &Function { decl, header_len, indent, asyncness } = self;
            let amp = if f.alternate() { "&" } else { "&amp;" };
            let mut args = String::new();
            let mut args_plain = String::new();
            for (i, input) in decl.inputs.values.iter().enumerate() {
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
                                args.push_str(&format!("self: {:#}", typ.print()));
                            } else {
                                args.push_str(&format!("self: {}", typ.print()));
                            }
                            args_plain.push_str(&format!("self: {:#}", typ.print()));
                        }
                    }
                } else {
                    if i > 0 {
                        args.push_str(" <br>");
                        args_plain.push(' ');
                    }
                    if !input.name.is_empty() {
                        args.push_str(&format!("{}: ", input.name));
                        args_plain.push_str(&format!("{}: ", input.name));
                    }

                    if f.alternate() {
                        args.push_str(&format!("{:#}", input.type_.print()));
                    } else {
                        args.push_str(&input.type_.print().to_string());
                    }
                    args_plain.push_str(&format!("{:#}", input.type_.print()));
                }
                if i + 1 < decl.inputs.values.len() {
                    args.push(',');
                    args_plain.push(',');
                }
            }

            let mut args_plain = format!("({})", args_plain);

            if decl.c_variadic {
                args.push_str(",<br> ...");
                args_plain.push_str(", ...");
            }

            let output = if let hir::IsAsync::Async = asyncness {
                Cow::Owned(decl.sugared_async_return_type())
            } else {
                Cow::Borrowed(&decl.output)
            };

            let arrow_plain = format!("{:#}", &output.print());
            let arrow = if f.alternate() {
                format!("{:#}", &output.print())
            } else {
                output.print().to_string()
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
        })
    }
}

impl clean::Visibility {
    crate fn print_with_space<'tcx>(
        self,
        tcx: TyCtxt<'tcx>,
        item_did: DefId,
    ) -> impl fmt::Display + 'tcx {
        use rustc_span::symbol::kw;

        display_fn(move |f| match self {
            clean::Public => f.write_str("pub "),
            clean::Inherited => Ok(()),

            clean::Visibility::Restricted(vis_did) => {
                // FIXME(camelid): This may not work correctly if `item_did` is a module.
                //                 However, rustdoc currently never displays a module's
                //                 visibility, so it shouldn't matter.
                let parent_module = find_nearest_parent_module(tcx, item_did);

                if vis_did.index == CRATE_DEF_INDEX {
                    write!(f, "pub(crate) ")
                } else if parent_module == Some(vis_did) {
                    // `pub(in foo)` where `foo` is the parent module
                    // is the same as no visibility modifier
                    Ok(())
                } else if parent_module
                    .map(|parent| find_nearest_parent_module(tcx, parent))
                    .flatten()
                    == Some(vis_did)
                {
                    write!(f, "pub(super) ")
                } else {
                    f.write_str("pub(")?;
                    let path = tcx.def_path(vis_did);
                    debug!("path={:?}", path);
                    let first_name =
                        path.data[0].data.get_opt_name().expect("modules are always named");
                    if path.data.len() != 1
                        || (first_name != kw::SelfLower && first_name != kw::Super)
                    {
                        f.write_str("in ")?;
                    }
                    // modified from `resolved_path()` to work with `DefPathData`
                    let last_name = path.data.last().unwrap().data.get_opt_name().unwrap();
                    for seg in &path.data[..path.data.len() - 1] {
                        write!(f, "{}::", seg.data.get_opt_name().unwrap())?;
                    }
                    let path = anchor(vis_did, &last_name.as_str()).to_string();
                    write!(f, "{}) ", path)
                }
            }
        })
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

impl PrintWithSpace for hir::Constness {
    fn print_with_space(&self) -> &str {
        match self {
            hir::Constness::Const => "const ",
            hir::Constness::NotConst => "",
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

impl clean::Import {
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| match self.kind {
            clean::ImportKind::Simple(name) => {
                if name == self.source.path.last() {
                    write!(f, "use {};", self.source.print())
                } else {
                    write!(f, "use {} as {};", self.source.print(), name)
                }
            }
            clean::ImportKind::Glob => {
                if self.source.path.segments.is_empty() {
                    write!(f, "use *;")
                } else {
                    write!(f, "use {}::*;", self.source.print())
                }
            }
        })
    }
}

impl clean::ImportSource {
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| match self.did {
            Some(did) => resolved_path(f, did, &self.path, true, false),
            _ => {
                for seg in &self.path.segments[..self.path.segments.len() - 1] {
                    write!(f, "{}::", seg.name)?;
                }
                let name = self.path.last_name();
                if let hir::def::Res::PrimTy(p) = self.path.res {
                    primitive_link(f, PrimitiveType::from(p), &*name)?;
                } else {
                    write!(f, "{}", name)?;
                }
                Ok(())
            }
        })
    }
}

impl clean::TypeBinding {
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| {
            f.write_str(&*self.name.as_str())?;
            match self.kind {
                clean::TypeBindingKind::Equality { ref ty } => {
                    if f.alternate() {
                        write!(f, " = {:#}", ty.print())?;
                    } else {
                        write!(f, " = {}", ty.print())?;
                    }
                }
                clean::TypeBindingKind::Constraint { ref bounds } => {
                    if !bounds.is_empty() {
                        if f.alternate() {
                            write!(f, ": {:#}", print_generic_bounds(bounds))?;
                        } else {
                            write!(f, ":&nbsp;{}", print_generic_bounds(bounds))?;
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
    crate fn print(&self) -> impl fmt::Display + '_ {
        display_fn(move |f| match self {
            clean::GenericArg::Lifetime(lt) => fmt::Display::fmt(&lt.print(), f),
            clean::GenericArg::Type(ty) => fmt::Display::fmt(&ty.print(), f),
            clean::GenericArg::Const(ct) => fmt::Display::fmt(&ct.print(), f),
        })
    }
}

crate fn display_fn(f: impl FnOnce(&mut fmt::Formatter<'_>) -> fmt::Result) -> impl fmt::Display {
    WithFormatter(Cell::new(Some(f)))
}

struct WithFormatter<F>(Cell<Option<F>>);

impl<F> fmt::Display for WithFormatter<F>
where
    F: FnOnce(&mut fmt::Formatter<'_>) -> fmt::Result,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self.0.take()).unwrap()(f)
    }
}
