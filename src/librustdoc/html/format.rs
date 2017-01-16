// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! HTML formatting module
//!
//! This module contains a large number of `fmt::Display` implementations for
//! various types in `rustdoc::clean`. These implementations all currently
//! assume that HTML output is desired, although it may be possible to redesign
//! them in the future to instead emit any format desired.

use std::fmt;
use std::iter::repeat;

use rustc::hir::def_id::DefId;
use syntax::abi::Abi;
use rustc::hir;

use clean::{self, PrimitiveType};
use core::DocAccessLevels;
use html::item_type::ItemType;
use html::escape::Escape;
use html::render;
use html::render::{cache, CURRENT_LOCATION_KEY};

/// Helper to render an optional visibility with a space after it (if the
/// visibility is preset)
#[derive(Copy, Clone)]
pub struct VisSpace<'a>(pub &'a Option<clean::Visibility>);
/// Similarly to VisSpace, this structure is used to render a function style with a
/// space after it.
#[derive(Copy, Clone)]
pub struct UnsafetySpace(pub hir::Unsafety);
/// Similarly to VisSpace, this structure is used to render a function constness
/// with a space after it.
#[derive(Copy, Clone)]
pub struct ConstnessSpace(pub hir::Constness);
/// Wrapper struct for properly emitting a method declaration.
pub struct Method<'a>(pub &'a clean::FnDecl, pub usize);
/// Similar to VisSpace, but used for mutability
#[derive(Copy, Clone)]
pub struct MutableSpace(pub clean::Mutability);
/// Similar to VisSpace, but used for mutability
#[derive(Copy, Clone)]
pub struct RawMutableSpace(pub clean::Mutability);
/// Wrapper struct for emitting a where clause from Generics.
pub struct WhereClause<'a>(pub &'a clean::Generics, pub usize);
/// Wrapper struct for emitting type parameter bounds.
pub struct TyParamBounds<'a>(pub &'a [clean::TyParamBound]);
/// Wrapper struct for emitting a comma-separated list of items
pub struct CommaSep<'a, T: 'a>(pub &'a [T]);
pub struct AbiSpace(pub Abi);

pub struct HRef<'a> {
    pub did: DefId,
    pub text: &'a str,
}

impl<'a> VisSpace<'a> {
    pub fn get(self) -> &'a Option<clean::Visibility> {
        let VisSpace(v) = self; v
    }
}

impl UnsafetySpace {
    pub fn get(&self) -> hir::Unsafety {
        let UnsafetySpace(v) = *self; v
    }
}

impl ConstnessSpace {
    pub fn get(&self) -> hir::Constness {
        let ConstnessSpace(v) = *self; v
    }
}

impl<'a, T: fmt::Display> fmt::Display for CommaSep<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, item) in self.0.iter().enumerate() {
            if i != 0 { write!(f, ", ")?; }
            fmt::Display::fmt(item, f)?;
        }
        Ok(())
    }
}

impl<'a> fmt::Display for TyParamBounds<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let &TyParamBounds(bounds) = self;
        for (i, bound) in bounds.iter().enumerate() {
            if i > 0 {
                f.write_str(" + ")?;
            }
            fmt::Display::fmt(bound, f)?;
        }
        Ok(())
    }
}

impl fmt::Display for clean::Generics {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.lifetimes.is_empty() && self.type_params.is_empty() { return Ok(()) }
        if f.alternate() {
            f.write_str("<")?;
        } else {
            f.write_str("&lt;")?;
        }

        for (i, life) in self.lifetimes.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", *life)?;
        }

        if !self.type_params.is_empty() {
            if !self.lifetimes.is_empty() {
                f.write_str(", ")?;
            }
            for (i, tp) in self.type_params.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?
                }
                f.write_str(&tp.name)?;

                if !tp.bounds.is_empty() {
                    if f.alternate() {
                        write!(f, ": {:#}", TyParamBounds(&tp.bounds))?;
                    } else {
                        write!(f, ":&nbsp;{}", TyParamBounds(&tp.bounds))?;
                    }
                }

                if let Some(ref ty) = tp.default {
                    if f.alternate() {
                        write!(f, " = {:#}", ty)?;
                    } else {
                        write!(f, "&nbsp;=&nbsp;{}", ty)?;
                    }
                };
            }
        }
        if f.alternate() {
            f.write_str(">")?;
        } else {
            f.write_str("&gt;")?;
        }
        Ok(())
    }
}

impl<'a> fmt::Display for WhereClause<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let &WhereClause(gens, pad) = self;
        if gens.where_predicates.is_empty() {
            return Ok(());
        }
        let mut clause = String::new();
        if f.alternate() {
            clause.push_str(" where ");
        } else {
            clause.push_str(" <span class='where'>where ");
        }
        for (i, pred) in gens.where_predicates.iter().enumerate() {
            if i > 0 {
                if f.alternate() {
                    clause.push_str(", ");
                } else {
                    clause.push_str(",<br>");
                }
            }
            match pred {
                &clean::WherePredicate::BoundPredicate { ref ty, ref bounds } => {
                    let bounds = bounds;
                    if f.alternate() {
                        clause.push_str(&format!("{:#}: {:#}", ty, TyParamBounds(bounds)));
                    } else {
                        clause.push_str(&format!("{}: {}", ty, TyParamBounds(bounds)));
                    }
                }
                &clean::WherePredicate::RegionPredicate { ref lifetime,
                                                          ref bounds } => {
                    clause.push_str(&format!("{}: ", lifetime));
                    for (i, lifetime) in bounds.iter().enumerate() {
                        if i > 0 {
                            clause.push_str(" + ");
                        }

                        clause.push_str(&format!("{}", lifetime));
                    }
                }
                &clean::WherePredicate::EqPredicate { ref lhs, ref rhs } => {
                    if f.alternate() {
                        clause.push_str(&format!("{:#} == {:#}", lhs, rhs));
                    } else {
                        clause.push_str(&format!("{} == {}", lhs, rhs));
                    }
                }
            }
        }
        if !f.alternate() {
            clause.push_str("</span>");
            let plain = format!("{:#}", self);
            if plain.len() + pad > 80 {
                //break it onto its own line regardless, but make sure method impls and trait
                //blocks keep their fixed padding (2 and 9, respectively)
                let padding = if pad > 10 {
                    clause = clause.replace("class='where'", "class='where fmt-newline'");
                    repeat("&nbsp;").take(8).collect::<String>()
                } else {
                    repeat("&nbsp;").take(pad + 6).collect::<String>()
                };
                clause = clause.replace("<br>", &format!("<br>{}", padding));
            } else {
                clause = clause.replace("<br>", " ");
            }
        }
        write!(f, "{}", clause)
    }
}

impl fmt::Display for clean::Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.get_ref())?;
        Ok(())
    }
}

impl fmt::Display for clean::PolyTrait {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.lifetimes.is_empty() {
            if f.alternate() {
                f.write_str("for<")?;
            } else {
                f.write_str("for&lt;")?;
            }
            for (i, lt) in self.lifetimes.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{}", lt)?;
            }
            if f.alternate() {
                f.write_str("> ")?;
            } else {
                f.write_str("&gt; ")?;
            }
        }
        if f.alternate() {
            write!(f, "{:#}", self.trait_)
        } else {
            write!(f, "{}", self.trait_)
        }
    }
}

impl fmt::Display for clean::TyParamBound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            clean::RegionBound(ref lt) => {
                write!(f, "{}", *lt)
            }
            clean::TraitBound(ref ty, modifier) => {
                let modifier_str = match modifier {
                    hir::TraitBoundModifier::None => "",
                    hir::TraitBoundModifier::Maybe => "?",
                };
                if f.alternate() {
                    write!(f, "{}{:#}", modifier_str, *ty)
                } else {
                    write!(f, "{}{}", modifier_str, *ty)
                }
            }
        }
    }
}

impl fmt::Display for clean::PathParameters {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            clean::PathParameters::AngleBracketed {
                ref lifetimes, ref types, ref bindings
            } => {
                if !lifetimes.is_empty() || !types.is_empty() || !bindings.is_empty() {
                    if f.alternate() {
                        f.write_str("<")?;
                    } else {
                        f.write_str("&lt;")?;
                    }
                    let mut comma = false;
                    for lifetime in lifetimes {
                        if comma {
                            f.write_str(", ")?;
                        }
                        comma = true;
                        write!(f, "{}", *lifetime)?;
                    }
                    for ty in types {
                        if comma {
                            f.write_str(", ")?;
                        }
                        comma = true;
                        if f.alternate() {
                            write!(f, "{:#}", *ty)?;
                        } else {
                            write!(f, "{}", *ty)?;
                        }
                    }
                    for binding in bindings {
                        if comma {
                            f.write_str(", ")?;
                        }
                        comma = true;
                        if f.alternate() {
                            write!(f, "{:#}", *binding)?;
                        } else {
                            write!(f, "{}", *binding)?;
                        }
                    }
                    if f.alternate() {
                        f.write_str(">")?;
                    } else {
                        f.write_str("&gt;")?;
                    }
                }
            }
            clean::PathParameters::Parenthesized { ref inputs, ref output } => {
                f.write_str("(")?;
                let mut comma = false;
                for ty in inputs {
                    if comma {
                        f.write_str(", ")?;
                    }
                    comma = true;
                    if f.alternate() {
                        write!(f, "{:#}", *ty)?;
                    } else {
                        write!(f, "{}", *ty)?;
                    }
                }
                f.write_str(")")?;
                if let Some(ref ty) = *output {
                    if f.alternate() {
                        write!(f, " -> {:#}", ty)?;
                    } else {
                        write!(f, " -&gt; {}", ty)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for clean::PathSegment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.name)?;
        if f.alternate() {
            write!(f, "{:#}", self.params)
        } else {
            write!(f, "{}", self.params)
        }
    }
}

impl fmt::Display for clean::Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.global {
            f.write_str("::")?
        }

        for (i, seg) in self.segments.iter().enumerate() {
            if i > 0 {
                f.write_str("::")?
            }
            if f.alternate() {
                write!(f, "{:#}", seg)?;
            } else {
                write!(f, "{}", seg)?;
            }
        }
        Ok(())
    }
}

pub fn href(did: DefId) -> Option<(String, ItemType, Vec<String>)> {
    let cache = cache();
    if !did.is_local() && !cache.access_levels.is_doc_reachable(did) {
        return None
    }

    let loc = CURRENT_LOCATION_KEY.with(|l| l.borrow().clone());
    let (fqp, shortty, mut url) = match cache.paths.get(&did) {
        Some(&(ref fqp, shortty)) => {
            (fqp, shortty, repeat("../").take(loc.len()).collect())
        }
        None => match cache.external_paths.get(&did) {
            Some(&(ref fqp, shortty)) => {
                (fqp, shortty, match cache.extern_locations[&did.krate] {
                    (.., render::Remote(ref s)) => s.to_string(),
                    (.., render::Local) => repeat("../").take(loc.len()).collect(),
                    (.., render::Unknown) => return None,
                })
            }
            None => return None,
        }
    };
    for component in &fqp[..fqp.len() - 1] {
        url.push_str(component);
        url.push_str("/");
    }
    match shortty {
        ItemType::Module => {
            url.push_str(fqp.last().unwrap());
            url.push_str("/index.html");
        }
        _ => {
            url.push_str(shortty.css_class());
            url.push_str(".");
            url.push_str(fqp.last().unwrap());
            url.push_str(".html");
        }
    }
    Some((url, shortty, fqp.to_vec()))
}

/// Used when rendering a `ResolvedPath` structure. This invokes the `path`
/// rendering function with the necessary arguments for linking to a local path.
fn resolved_path(w: &mut fmt::Formatter, did: DefId, path: &clean::Path,
                 print_all: bool, use_absolute: bool) -> fmt::Result {
    let last = path.segments.last().unwrap();
    let rel_root = match &*path.segments[0].name {
        "self" => Some("./".to_string()),
        _ => None,
    };

    if print_all {
        let amt = path.segments.len() - 1;
        match rel_root {
            Some(mut root) => {
                for seg in &path.segments[..amt] {
                    if "super" == seg.name || "self" == seg.name || w.alternate() {
                        write!(w, "{}::", seg.name)?;
                    } else {
                        root.push_str(&seg.name);
                        root.push_str("/");
                        write!(w, "<a class='mod'
                                       href='{}index.html'>{}</a>::",
                                 root,
                                 seg.name)?;
                    }
                }
            }
            None => {
                for seg in &path.segments[..amt] {
                    write!(w, "{}::", seg.name)?;
                }
            }
        }
    }
    if w.alternate() {
        write!(w, "{:#}{:#}", HRef::new(did, &last.name), last.params)?;
    } else {
        let path = if use_absolute {
            match href(did) {
                Some((_, _, fqp)) => format!("{}::{}",
                                             fqp[..fqp.len()-1].join("::"),
                                             HRef::new(did, fqp.last().unwrap())),
                None => format!("{}", HRef::new(did, &last.name)),
            }
        } else {
            format!("{}", HRef::new(did, &last.name))
        };
        write!(w, "{}{}", path, last.params)?;
    }
    Ok(())
}

fn primitive_link(f: &mut fmt::Formatter,
                  prim: clean::PrimitiveType,
                  name: &str) -> fmt::Result {
    let m = cache();
    let mut needs_termination = false;
    if !f.alternate() {
        match m.primitive_locations.get(&prim) {
            Some(&def_id) if def_id.is_local() => {
                let len = CURRENT_LOCATION_KEY.with(|s| s.borrow().len());
                let len = if len == 0 {0} else {len - 1};
                write!(f, "<a class='primitive' href='{}primitive.{}.html'>",
                       repeat("../").take(len).collect::<String>(),
                       prim.to_url_str())?;
                needs_termination = true;
            }
            Some(&def_id) => {
                let loc = match m.extern_locations[&def_id.krate] {
                    (ref cname, _, render::Remote(ref s)) => {
                        Some((cname, s.to_string()))
                    }
                    (ref cname, _, render::Local) => {
                        let len = CURRENT_LOCATION_KEY.with(|s| s.borrow().len());
                        Some((cname, repeat("../").take(len).collect::<String>()))
                    }
                    (.., render::Unknown) => None,
                };
                if let Some((cname, root)) = loc {
                    write!(f, "<a class='primitive' href='{}{}/primitive.{}.html'>",
                           root,
                           cname,
                           prim.to_url_str())?;
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
fn tybounds(w: &mut fmt::Formatter,
            typarams: &Option<Vec<clean::TyParamBound> >) -> fmt::Result {
    match *typarams {
        Some(ref params) => {
            for param in params {
                write!(w, " + ")?;
                fmt::Display::fmt(param, w)?;
            }
            Ok(())
        }
        None => Ok(())
    }
}

impl<'a> HRef<'a> {
    pub fn new(did: DefId, text: &'a str) -> HRef<'a> {
        HRef { did: did, text: text }
    }
}

impl<'a> fmt::Display for HRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match href(self.did) {
            Some((url, shortty, fqp)) => if !f.alternate() {
                write!(f, "<a class='{}' href='{}' title='{}'>{}</a>",
                       shortty, url, fqp.join("::"), self.text)
            } else {
                write!(f, "{}", self.text)
            },
            _ => write!(f, "{}", self.text),
        }
    }
}

fn fmt_type(t: &clean::Type, f: &mut fmt::Formatter, use_absolute: bool) -> fmt::Result {
    match *t {
        clean::Generic(ref name) => {
            f.write_str(name)
        }
        clean::ResolvedPath{ did, ref typarams, ref path, is_generic } => {
            // Paths like T::Output and Self::Output should be rendered with all segments
            resolved_path(f, did, path, is_generic, use_absolute)?;
            tybounds(f, typarams)
        }
        clean::Infer => write!(f, "_"),
        clean::Primitive(prim) => primitive_link(f, prim, prim.as_str()),
        clean::BareFunction(ref decl) => {
            if f.alternate() {
                write!(f, "{}{}fn{:#}{:#}",
                       UnsafetySpace(decl.unsafety),
                       AbiSpace(decl.abi),
                       decl.generics,
                       decl.decl)
            } else {
                write!(f, "{}{}fn{}{}",
                       UnsafetySpace(decl.unsafety),
                       AbiSpace(decl.abi),
                       decl.generics,
                       decl.decl)
            }
        }
        clean::Tuple(ref typs) => {
            match &typs[..] {
                &[] => primitive_link(f, PrimitiveType::Tuple, "()"),
                &[ref one] => {
                    primitive_link(f, PrimitiveType::Tuple, "(")?;
                    //carry f.alternate() into this display w/o branching manually
                    fmt::Display::fmt(one, f)?;
                    primitive_link(f, PrimitiveType::Tuple, ",)")
                }
                many => {
                    primitive_link(f, PrimitiveType::Tuple, "(")?;
                    fmt::Display::fmt(&CommaSep(&many), f)?;
                    primitive_link(f, PrimitiveType::Tuple, ")")
                }
            }
        }
        clean::Vector(ref t) => {
            primitive_link(f, PrimitiveType::Slice, &format!("["))?;
            fmt::Display::fmt(t, f)?;
            primitive_link(f, PrimitiveType::Slice, &format!("]"))
        }
        clean::FixedVector(ref t, ref s) => {
            primitive_link(f, PrimitiveType::Array, "[")?;
            fmt::Display::fmt(t, f)?;
            if f.alternate() {
                primitive_link(f, PrimitiveType::Array,
                               &format!("; {}]", s))
            } else {
                primitive_link(f, PrimitiveType::Array,
                               &format!("; {}]", Escape(s)))
            }
        }
        clean::Never => f.write_str("!"),
        clean::RawPointer(m, ref t) => {
            match **t {
                clean::Generic(_) | clean::ResolvedPath {is_generic: true, ..} => {
                    if f.alternate() {
                        primitive_link(f, clean::PrimitiveType::RawPointer,
                                       &format!("*{}{:#}", RawMutableSpace(m), t))
                    } else {
                        primitive_link(f, clean::PrimitiveType::RawPointer,
                                       &format!("*{}{}", RawMutableSpace(m), t))
                    }
                }
                _ => {
                    primitive_link(f, clean::PrimitiveType::RawPointer,
                                   &format!("*{}", RawMutableSpace(m)))?;
                    fmt::Display::fmt(t, f)
                }
            }
        }
        clean::BorrowedRef{ lifetime: ref l, mutability, type_: ref ty} => {
            let lt = match *l {
                Some(ref l) => format!("{} ", *l),
                _ => "".to_string(),
            };
            let m = MutableSpace(mutability);
            match **ty {
                clean::Vector(ref bt) => { // BorrowedRef{ ... Vector(T) } is &[T]
                    match **bt {
                        clean::Generic(_) =>
                            if f.alternate() {
                                primitive_link(f, PrimitiveType::Slice,
                                    &format!("&{}{}[{:#}]", lt, m, **bt))
                            } else {
                                primitive_link(f, PrimitiveType::Slice,
                                    &format!("&amp;{}{}[{}]", lt, m, **bt))
                            },
                        _ => {
                            if f.alternate() {
                                primitive_link(f, PrimitiveType::Slice,
                                               &format!("&{}{}[", lt, m))?;
                                write!(f, "{:#}", **bt)?;
                            } else {
                                primitive_link(f, PrimitiveType::Slice,
                                               &format!("&amp;{}{}[", lt, m))?;
                                write!(f, "{}", **bt)?;
                            }
                            primitive_link(f, PrimitiveType::Slice, "]")
                        }
                    }
                }
                _ => {
                    if f.alternate() {
                        write!(f, "&{}{}{:#}", lt, m, **ty)
                    } else {
                        write!(f, "&amp;{}{}{}", lt, m, **ty)
                    }
                }
            }
        }
        clean::ImplTrait(ref bounds) => {
            write!(f, "impl ")?;
            for (i, bound) in bounds.iter().enumerate() {
                if i != 0 {
                    write!(f, " + ")?;
                }
                if f.alternate() {
                    write!(f, "{:#}", *bound)?;
                } else {
                    write!(f, "{}", *bound)?;
                }
            }
            Ok(())
        }
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
        clean::QPath {
            ref name,
            ref self_type,
            trait_: box clean::ResolvedPath { did, ref typarams, .. },
        } => {
            if f.alternate() {
                write!(f, "{:#}::", self_type)?;
            } else {
                write!(f, "{}::", self_type)?;
            }
            let path = clean::Path::singleton(name.clone());
            resolved_path(f, did, &path, true, use_absolute)?;

            // FIXME: `typarams` are not rendered, and this seems bad?
            drop(typarams);
            Ok(())
        }
        clean::QPath { ref name, ref self_type, ref trait_ } => {
            if f.alternate() {
                write!(f, "<{:#} as {:#}>::{}", self_type, trait_, name)
            } else {
                write!(f, "&lt;{} as {}&gt;::{}", self_type, trait_, name)
            }
        }
        clean::Unique(..) => {
            panic!("should have been cleaned")
        }
    }
}

impl fmt::Display for clean::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt_type(self, f, false)
    }
}

fn fmt_impl(i: &clean::Impl,
            f: &mut fmt::Formatter,
            link_trait: bool,
            use_absolute: bool) -> fmt::Result {
    let mut plain = String::new();

    if f.alternate() {
        write!(f, "impl{:#} ", i.generics)?;
    } else {
        write!(f, "impl{} ", i.generics)?;
    }
    plain.push_str(&format!("impl{:#} ", i.generics));

    if let Some(ref ty) = i.trait_ {
        if i.polarity == Some(clean::ImplPolarity::Negative) {
            write!(f, "!")?;
            plain.push_str("!");
        }

        if link_trait {
            fmt::Display::fmt(ty, f)?;
            plain.push_str(&format!("{:#}", ty));
        } else {
            match *ty {
                clean::ResolvedPath { typarams: None, ref path, is_generic: false, .. } => {
                    let last = path.segments.last().unwrap();
                    fmt::Display::fmt(&last.name, f)?;
                    fmt::Display::fmt(&last.params, f)?;
                    plain.push_str(&format!("{:#}{:#}", last.name, last.params));
                }
                _ => unreachable!(),
            }
        }
        write!(f, " for ")?;
        plain.push_str(" for ");
    }

    fmt_type(&i.for_, f, use_absolute)?;
    plain.push_str(&format!("{:#}", i.for_));

    fmt::Display::fmt(&WhereClause(&i.generics, plain.len() + 1), f)?;
    Ok(())
}

impl fmt::Display for clean::Impl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt_impl(self, f, true, false)
    }
}

// The difference from above is that trait is not hyperlinked.
pub fn fmt_impl_for_trait_page(i: &clean::Impl,
                               f: &mut fmt::Formatter,
                               use_absolute: bool) -> fmt::Result {
    fmt_impl(i, f, false, use_absolute)
}

impl fmt::Display for clean::Arguments {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, input) in self.values.iter().enumerate() {
            if !input.name.is_empty() {
                write!(f, "{}: ", input.name)?;
            }
            if f.alternate() {
                write!(f, "{:#}", input.type_)?;
            } else {
                write!(f, "{}", input.type_)?;
            }
            if i + 1 < self.values.len() { write!(f, ", ")?; }
        }
        Ok(())
    }
}

impl fmt::Display for clean::FunctionRetTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            clean::Return(clean::Tuple(ref tys)) if tys.is_empty() => Ok(()),
            clean::Return(ref ty) if f.alternate() => write!(f, " -> {:#}", ty),
            clean::Return(ref ty) => write!(f, " -&gt; {}", ty),
            clean::DefaultReturn => Ok(()),
        }
    }
}

impl fmt::Display for clean::FnDecl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.variadic {
            if f.alternate() {
                write!(f, "({args:#}, ...){arrow:#}", args = self.inputs, arrow = self.output)
            } else {
                write!(f, "({args}, ...){arrow}", args = self.inputs, arrow = self.output)
            }
        } else {
            if f.alternate() {
                write!(f, "({args:#}){arrow:#}", args = self.inputs, arrow = self.output)
            } else {
                write!(f, "({args}){arrow}", args = self.inputs, arrow = self.output)
            }
        }
    }
}

impl<'a> fmt::Display for Method<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let decl = self.0;
        let indent = self.1;
        let amp = if f.alternate() { "&" } else { "&amp;" };
        let mut args = String::new();
        let mut args_plain = String::new();
        for (i, input) in decl.inputs.values.iter().enumerate() {
            if let Some(selfty) = input.to_self() {
                match selfty {
                    clean::SelfValue => {
                        args.push_str("self");
                        args_plain.push_str("self");
                    }
                    clean::SelfBorrowed(Some(ref lt), mtbl) => {
                        args.push_str(&format!("{}{} {}self", amp, *lt, MutableSpace(mtbl)));
                        args_plain.push_str(&format!("&{} {}self", *lt, MutableSpace(mtbl)));
                    }
                    clean::SelfBorrowed(None, mtbl) => {
                        args.push_str(&format!("{}{}self", amp, MutableSpace(mtbl)));
                        args_plain.push_str(&format!("&{}self", MutableSpace(mtbl)));
                    }
                    clean::SelfExplicit(ref typ) => {
                        if f.alternate() {
                            args.push_str(&format!("self: {:#}", *typ));
                        } else {
                            args.push_str(&format!("self: {}", *typ));
                        }
                        args_plain.push_str(&format!("self: {:#}", *typ));
                    }
                }
            } else {
                if i > 0 {
                    args.push_str("<br> ");
                    args_plain.push_str(" ");
                }
                if !input.name.is_empty() {
                    args.push_str(&format!("{}: ", input.name));
                    args_plain.push_str(&format!("{}: ", input.name));
                }

                if f.alternate() {
                    args.push_str(&format!("{:#}", input.type_));
                } else {
                    args.push_str(&format!("{}", input.type_));
                }
                args_plain.push_str(&format!("{:#}", input.type_));
            }
            if i + 1 < decl.inputs.values.len() {
                args.push_str(",");
                args_plain.push_str(",");
            }
        }

        if decl.variadic {
            args.push_str(",<br> ...");
            args_plain.push_str(", ...");
        }

        let arrow_plain = format!("{:#}", decl.output);
        let arrow = if f.alternate() {
            format!("{:#}", decl.output)
        } else {
            format!("{}", decl.output)
        };

        let mut output: String;
        let plain: String;
        let pad = repeat(" ").take(indent).collect::<String>();
        if arrow.is_empty() {
            output = format!("({})", args);
            plain = format!("{}({})", pad, args_plain);
        } else {
            output = format!("({args})<br>{arrow}", args = args, arrow = arrow);
            plain = format!("{pad}({args}){arrow}",
                            pad = pad,
                            args = args_plain,
                            arrow = arrow_plain);
        }

        if plain.len() > 80 {
            let pad = repeat("&nbsp;").take(indent).collect::<String>();
            let pad = format!("<br>{}", pad);
            output = output.replace("<br>", &pad);
        } else {
            output = output.replace("<br>", "");
        }
        if f.alternate() {
            write!(f, "{}", output.replace("<br>", "\n"))
        } else {
            write!(f, "{}", output)
        }
    }
}

impl<'a> fmt::Display for VisSpace<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self.get() {
            Some(clean::Public) => write!(f, "pub "),
            Some(clean::Inherited) | None => Ok(())
        }
    }
}

impl fmt::Display for UnsafetySpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.get() {
            hir::Unsafety::Unsafe => write!(f, "unsafe "),
            hir::Unsafety::Normal => Ok(())
        }
    }
}

impl fmt::Display for ConstnessSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.get() {
            hir::Constness::Const => write!(f, "const "),
            hir::Constness::NotConst => Ok(())
        }
    }
}

impl fmt::Display for clean::Import {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            clean::Import::Simple(ref name, ref src) => {
                if *name == src.path.last_name() {
                    write!(f, "use {};", *src)
                } else {
                    write!(f, "use {} as {};", *src, *name)
                }
            }
            clean::Import::Glob(ref src) => {
                write!(f, "use {}::*;", *src)
            }
        }
    }
}

impl fmt::Display for clean::ImportSource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.did {
            Some(did) => resolved_path(f, did, &self.path, true, false),
            _ => {
                for (i, seg) in self.path.segments.iter().enumerate() {
                    if i > 0 {
                        write!(f, "::")?
                    }
                    write!(f, "{}", seg.name)?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for clean::TypeBinding {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if f.alternate() {
            write!(f, "{}={:#}", self.name, self.ty)
        } else {
            write!(f, "{}={}", self.name, self.ty)
        }
    }
}

impl fmt::Display for MutableSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            MutableSpace(clean::Immutable) => Ok(()),
            MutableSpace(clean::Mutable) => write!(f, "mut "),
        }
    }
}

impl fmt::Display for RawMutableSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            RawMutableSpace(clean::Immutable) => write!(f, "const "),
            RawMutableSpace(clean::Mutable) => write!(f, "mut "),
        }
    }
}

impl fmt::Display for AbiSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let quot = if f.alternate() { "\"" } else { "&quot;" };
        match self.0 {
            Abi::Rust => Ok(()),
            Abi::C => write!(f, "extern "),
            abi => write!(f, "extern {0}{1}{0} ", quot, abi.name()),
        }
    }
}
