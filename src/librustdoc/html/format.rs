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

use rustc::middle::cstore::LOCAL_CRATE;
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
use syntax::abi::Abi;
use rustc::hir;

use clean;
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
pub struct Method<'a>(pub &'a clean::FnDecl);
/// Similar to VisSpace, but used for mutability
#[derive(Copy, Clone)]
pub struct MutableSpace(pub clean::Mutability);
/// Similar to VisSpace, but used for mutability
#[derive(Copy, Clone)]
pub struct RawMutableSpace(pub clean::Mutability);
/// Wrapper struct for emitting a where clause from Generics.
pub struct WhereClause<'a>(pub &'a clean::Generics);
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
            write!(f, "{}", item)?;
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
            write!(f, "{}", *bound)?;
        }
        Ok(())
    }
}

impl fmt::Display for clean::Generics {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.lifetimes.is_empty() && self.type_params.is_empty() { return Ok(()) }
        f.write_str("&lt;")?;

        for (i, life) in self.lifetimes.iter().enumerate() {
            if i > 0 {
                f.write_str(",&nbsp;")?;
            }
            write!(f, "{}", *life)?;
        }

        if !self.type_params.is_empty() {
            if !self.lifetimes.is_empty() {
                f.write_str(",&nbsp;")?;
            }
            for (i, tp) in self.type_params.iter().enumerate() {
                if i > 0 {
                    f.write_str(",&nbsp;")?
                }
                f.write_str(&tp.name)?;

                if !tp.bounds.is_empty() {
                    write!(f, ":&nbsp;{}", TyParamBounds(&tp.bounds))?;
                }

                match tp.default {
                    Some(ref ty) => { write!(f, "&nbsp;=&nbsp;{}", ty)?; },
                    None => {}
                };
            }
        }
        f.write_str("&gt;")?;
        Ok(())
    }
}

impl<'a> fmt::Display for WhereClause<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let &WhereClause(gens) = self;
        if gens.where_predicates.is_empty() {
            return Ok(());
        }
        f.write_str(" <span class='where'>where ")?;
        for (i, pred) in gens.where_predicates.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            match pred {
                &clean::WherePredicate::BoundPredicate { ref ty, ref bounds } => {
                    let bounds = bounds;
                    write!(f, "{}: {}", ty, TyParamBounds(bounds))?;
                }
                &clean::WherePredicate::RegionPredicate { ref lifetime,
                                                          ref bounds } => {
                    write!(f, "{}: ", lifetime)?;
                    for (i, lifetime) in bounds.iter().enumerate() {
                        if i > 0 {
                            f.write_str(" + ")?;
                        }

                        write!(f, "{}", lifetime)?;
                    }
                }
                &clean::WherePredicate::EqPredicate { ref lhs, ref rhs } => {
                    write!(f, "{} == {}", lhs, rhs)?;
                }
            }
        }
        f.write_str("</span>")?;
        Ok(())
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
            f.write_str("for&lt;")?;
            for (i, lt) in self.lifetimes.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{}", lt)?;
            }
            f.write_str("&gt; ")?;
        }
        write!(f, "{}", self.trait_)
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
                write!(f, "{}{}", modifier_str, *ty)
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
                    f.write_str("&lt;")?;
                    let mut comma = false;
                    for lifetime in lifetimes {
                        if comma {
                            f.write_str(",&nbsp;")?;
                        }
                        comma = true;
                        write!(f, "{}", *lifetime)?;
                    }
                    for ty in types {
                        if comma {
                            f.write_str(",&nbsp;")?;
                        }
                        comma = true;
                        write!(f, "{}", *ty)?;
                    }
                    for binding in bindings {
                        if comma {
                            f.write_str(",&nbsp;")?;
                        }
                        comma = true;
                        write!(f, "{}", *binding)?;
                    }
                    f.write_str("&gt;")?;
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
                    write!(f, "{}", *ty)?;
                }
                f.write_str(")")?;
                if let Some(ref ty) = *output {
                    f.write_str(" -&gt; ")?;
                    write!(f, "{}", ty)?;
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for clean::PathSegment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.name)?;
        write!(f, "{}", self.params)
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
            write!(f, "{}", seg)?;
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
    let &(ref fqp, shortty) = match cache.paths.get(&did) {
        Some(p) => p,
        None => return None,
    };

    let mut url = if did.is_local() || cache.inlined.contains(&did) {
        repeat("../").take(loc.len()).collect::<String>()
    } else {
        match cache.extern_locations[&did.krate] {
            (_, render::Remote(ref s)) => s.to_string(),
            (_, render::Local) => repeat("../").take(loc.len()).collect(),
            (_, render::Unknown) => return None,
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
            url.push_str(shortty.to_static_str());
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
                 print_all: bool) -> fmt::Result {
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
                    if "super" == seg.name || "self" == seg.name {
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
    write!(w, "{}{}", HRef::new(did, &last.name), last.params)?;
    Ok(())
}

fn primitive_link(f: &mut fmt::Formatter,
                  prim: clean::PrimitiveType,
                  name: &str) -> fmt::Result {
    let m = cache();
    let mut needs_termination = false;
    match m.primitive_locations.get(&prim) {
        Some(&LOCAL_CRATE) => {
            let len = CURRENT_LOCATION_KEY.with(|s| s.borrow().len());
            let len = if len == 0 {0} else {len - 1};
            write!(f, "<a class='primitive' href='{}primitive.{}.html'>",
                   repeat("../").take(len).collect::<String>(),
                   prim.to_url_str())?;
            needs_termination = true;
        }
        Some(&cnum) => {
            let path = &m.paths[&DefId {
                krate: cnum,
                index: CRATE_DEF_INDEX,
            }];
            let loc = match m.extern_locations[&cnum] {
                (_, render::Remote(ref s)) => Some(s.to_string()),
                (_, render::Local) => {
                    let len = CURRENT_LOCATION_KEY.with(|s| s.borrow().len());
                    Some(repeat("../").take(len).collect::<String>())
                }
                (_, render::Unknown) => None,
            };
            match loc {
                Some(root) => {
                    write!(f, "<a class='primitive' href='{}{}/primitive.{}.html'>",
                           root,
                           path.0.first().unwrap(),
                           prim.to_url_str())?;
                    needs_termination = true;
                }
                None => {}
            }
        }
        None => {}
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
                write!(w, "{}", *param)?;
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
            Some((url, shortty, fqp)) => {
                write!(f, "<a class='{}' href='{}' title='{}'>{}</a>",
                       shortty, url, fqp.join("::"), self.text)
            }
            _ => write!(f, "{}", self.text),
        }
    }
}

impl fmt::Display for clean::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            clean::Generic(ref name) => {
                f.write_str(name)
            }
            clean::ResolvedPath{ did, ref typarams, ref path, is_generic } => {
                // Paths like T::Output and Self::Output should be rendered with all segments
                resolved_path(f, did, path, is_generic)?;
                tybounds(f, typarams)
            }
            clean::Infer => write!(f, "_"),
            clean::Primitive(prim) => primitive_link(f, prim, prim.to_string()),
            clean::BareFunction(ref decl) => {
                write!(f, "{}{}fn{}{}",
                       UnsafetySpace(decl.unsafety),
                       AbiSpace(decl.abi),
                       decl.generics,
                       decl.decl)
            }
            clean::Tuple(ref typs) => {
                match &**typs {
                    [] => primitive_link(f, clean::PrimitiveTuple, "()"),
                    [ref one] => {
                        primitive_link(f, clean::PrimitiveTuple, "(")?;
                        write!(f, "{},", one)?;
                        primitive_link(f, clean::PrimitiveTuple, ")")
                    }
                    many => {
                        primitive_link(f, clean::PrimitiveTuple, "(")?;
                        write!(f, "{}", CommaSep(&many))?;
                        primitive_link(f, clean::PrimitiveTuple, ")")
                    }
                }
            }
            clean::Vector(ref t) => {
                primitive_link(f, clean::Slice, &format!("["))?;
                write!(f, "{}", t)?;
                primitive_link(f, clean::Slice, &format!("]"))
            }
            clean::FixedVector(ref t, ref s) => {
                primitive_link(f, clean::PrimitiveType::Array, "[")?;
                write!(f, "{}", t)?;
                primitive_link(f, clean::PrimitiveType::Array,
                               &format!("; {}]", Escape(s)))
            }
            clean::Bottom => f.write_str("!"),
            clean::RawPointer(m, ref t) => {
                match **t {
                    clean::Generic(_) | clean::ResolvedPath {is_generic: true, ..} => {
                        primitive_link(f, clean::PrimitiveType::PrimitiveRawPointer,
                                       &format!("*{}{}", RawMutableSpace(m), t))
                    }
                    _ => {
                        primitive_link(f, clean::PrimitiveType::PrimitiveRawPointer,
                                       &format!("*{}", RawMutableSpace(m)))?;
                        write!(f, "{}", t)
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
                                primitive_link(f, clean::Slice,
                                    &format!("&amp;{}{}[{}]", lt, m, **bt)),
                            _ => {
                                primitive_link(f, clean::Slice, &format!("&amp;{}{}[", lt, m))?;
                                write!(f, "{}", **bt)?;
                                primitive_link(f, clean::Slice, "]")
                            }
                        }
                    }
                    _ => {
                        write!(f, "&amp;{}{}{}", lt, m, **ty)
                    }
                }
            }
            clean::PolyTraitRef(ref bounds) => {
                for (i, bound) in bounds.iter().enumerate() {
                    if i != 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{}", *bound)?;
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
                write!(f, "{}::", self_type)?;
                let path = clean::Path::singleton(name.clone());
                resolved_path(f, did, &path, false)?;

                // FIXME: `typarams` are not rendered, and this seems bad?
                drop(typarams);
                Ok(())
            }
            clean::QPath { ref name, ref self_type, ref trait_ } => {
                write!(f, "&lt;{} as {}&gt;::{}", self_type, trait_, name)
            }
            clean::Unique(..) => {
                panic!("should have been cleaned")
            }
        }
    }
}

fn fmt_impl(i: &clean::Impl, f: &mut fmt::Formatter, link_trait: bool) -> fmt::Result {
    write!(f, "impl{} ", i.generics)?;
    if let Some(ref ty) = i.trait_ {
        write!(f, "{}",
               if i.polarity == Some(clean::ImplPolarity::Negative) { "!" } else { "" })?;
        if link_trait {
            write!(f, "{}", *ty)?;
        } else {
            match *ty {
                clean::ResolvedPath{ typarams: None, ref path, is_generic: false, .. } => {
                    let last = path.segments.last().unwrap();
                    write!(f, "{}{}", last.name, last.params)?;
                }
                _ => unreachable!(),
            }
        }
        write!(f, " for ")?;
    }
    write!(f, "{}{}", i.for_, WhereClause(&i.generics))?;
    Ok(())
}

impl fmt::Display for clean::Impl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt_impl(self, f, true)
    }
}

// The difference from above is that trait is not hyperlinked.
pub fn fmt_impl_for_trait_page(i: &clean::Impl, f: &mut fmt::Formatter) -> fmt::Result {
    fmt_impl(i, f, false)
}

impl fmt::Display for clean::Arguments {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, input) in self.values.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            if !input.name.is_empty() {
                write!(f, "{}: ", input.name)?;
            }
            write!(f, "{}", input.type_)?;
        }
        Ok(())
    }
}

impl fmt::Display for clean::FunctionRetTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            clean::Return(clean::Tuple(ref tys)) if tys.is_empty() => Ok(()),
            clean::Return(ref ty) => write!(f, " -&gt; {}", ty),
            clean::DefaultReturn => Ok(()),
            clean::NoReturn => write!(f, " -&gt; !")
        }
    }
}

impl fmt::Display for clean::FnDecl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.variadic {
            write!(f, "({args}, ...){arrow}", args = self.inputs, arrow = self.output)
        } else {
            write!(f, "({args}){arrow}", args = self.inputs, arrow = self.output)
        }
    }
}

impl<'a> fmt::Display for Method<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let decl = self.0;
        let mut args = String::new();
        for (i, input) in decl.inputs.values.iter().enumerate() {
            if i > 0 || !args.is_empty() { args.push_str(", "); }
            if let Some(selfty) = input.to_self() {
                match selfty {
                    clean::SelfValue => args.push_str("self"),
                    clean::SelfBorrowed(Some(ref lt), mtbl) => {
                        args.push_str(&format!("&amp;{} {}self", *lt, MutableSpace(mtbl)));
                    }
                    clean::SelfBorrowed(None, mtbl) => {
                        args.push_str(&format!("&amp;{}self", MutableSpace(mtbl)));
                    }
                    clean::SelfExplicit(ref typ) => {
                        args.push_str(&format!("self: {}", *typ));
                    }
                }
            } else {
                if !input.name.is_empty() {
                    args.push_str(&format!("{}: ", input.name));
                }
                args.push_str(&format!("{}", input.type_));
            }
        }
        write!(f, "({args}){arrow}", args = args, arrow = decl.output)
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
            clean::SimpleImport(ref name, ref src) => {
                if *name == src.path.last_name() {
                    write!(f, "use {};", *src)
                } else {
                    write!(f, "use {} as {};", *src, *name)
                }
            }
            clean::GlobImport(ref src) => {
                write!(f, "use {}::*;", *src)
            }
            clean::ImportList(ref src, ref names) => {
                write!(f, "use {}::{{", *src)?;
                for (i, n) in names.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", *n)?;
                }
                write!(f, "}};")
            }
        }
    }
}

impl fmt::Display for clean::ImportSource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.did {
            Some(did) => resolved_path(f, did, &self.path, true),
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

impl fmt::Display for clean::ViewListIdent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.source {
            Some(did) => {
                let path = clean::Path::singleton(self.name.clone());
                resolved_path(f, did, &path, false)?;
            }
            _ => write!(f, "{}", self.name)?,
        }

        if let Some(ref name) = self.rename {
            write!(f, " as {}", name)?;
        }
        Ok(())
    }
}

impl fmt::Display for clean::TypeBinding {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}={}", self.name, self.ty)
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
        match self.0 {
            Abi::Rust => Ok(()),
            Abi::C => write!(f, "extern "),
            abi => write!(f, "extern &quot;{}&quot; ", abi.name()),
        }
    }
}
