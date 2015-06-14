// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Ascii formatting module
//!
//! This module contains a large number of `fmt::Display` implementations for
//! various types in `rustdoc::clean`. These implementations all currently
//! assume that HTML output is desired, although it may be possible to redesign
//! them in the future to instead emit any format desired.

extern crate term;

use std::fmt;
use std::iter::repeat;

use syntax::abi::Abi;
use syntax::ast;
use syntax::ast_util;

use clean;
use ascii::item_type::ItemType;
use ascii::render;
use ascii::render::{cache, CURRENT_LOCATION_KEY};

const RED : &'static str = "\x1b[31m";
const GREEN : &'static str = "\x1b[32m";
const BROWN : &'static str = "\x1b[33m";
const BLUE : &'static str = "\x1b[34m";
const MAGENTA : &'static str = "\x1b[35m";
const CYAN : &'static str = "\x1b[36m";
const WHITE : &'static str = "\x1b[37m";
const NORMAL : &'static str = "\x1b[0m";

pub trait AsciiDisplay {
    fn get_display(&self) -> String;
}

/// Helper to render an optional visibility with a space after it (if the
/// visibility is preset)
#[derive(Copy, Clone)]
pub struct VisSpace(pub Option<ast::Visibility>);
/// Similarly to VisSpace, this structure is used to render a function style with a
/// space after it.
#[derive(Copy, Clone)]
pub struct UnsafetySpace(pub ast::Unsafety);
/// Similarly to VisSpace, this structure is used to render a function constness
/// with a space after it.
#[derive(Copy, Clone)]
pub struct ConstnessSpace(pub ast::Constness);
/// Wrapper struct for properly emitting a method declaration.
pub struct Method<'a>(pub &'a clean::SelfTy, pub &'a clean::FnDecl);
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

pub fn convert_color(ty_: &str) {
    let known = vec!("u8", "u16", "u32", "u64");
}

impl VisSpace {
    pub fn get(&self) -> Option<ast::Visibility> {
        let VisSpace(v) = *self; v
    }
}

impl UnsafetySpace {
    pub fn get(&self) -> ast::Unsafety {
        let UnsafetySpace(v) = *self; v
    }
}

impl ConstnessSpace {
    pub fn get(&self) -> ast::Constness {
        let ConstnessSpace(v) = *self; v
    }
}

impl<'a, T: fmt::Display> fmt::Display for CommaSep<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, item) in self.0.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}", item));
        }
        Ok(())
    }
}

impl<'a> fmt::Display for TyParamBounds<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let &TyParamBounds(bounds) = self;
        for (i, bound) in bounds.iter().enumerate() {
            if i > 0 {
                try!(f.write_str(" + "));
            }
            try!(write!(f, "{}", *bound));
        }
        Ok(())
    }
}

/*impl fmt::Display for clean::Generics {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.lifetimes.is_empty() && self.type_params.is_empty() { return Ok(()) }
        try!(f.write_str("<"));

        for (i, life) in self.lifetimes.iter().enumerate() {
            if i > 0 {
                try!(f.write_str(", "));
            }
            try!(write!(f, "{}", *life));
        }

        if !self.type_params.is_empty() {
            if !self.lifetimes.is_empty() {
                try!(f.write_str(", "));
            }
            for (i, tp) in self.type_params.iter().enumerate() {
                if i > 0 {
                    try!(f.write_str(", "))
                }
                try!(f.write_str(&tp.name));

                if !tp.bounds.is_empty() {
                    try!(write!(f, ": {}", TyParamBounds(&tp.bounds)));
                }

                match tp.default {
                    Some(ref ty) => { try!(write!(f, " = {}", ty)); },
                    None => {}
                };
            }
        }
        try!(f.write_str(">"));
        Ok(())
    }
}*/

impl AsciiDisplay for clean::Generics {
    fn get_display(&self) -> String {
        if self.lifetimes.is_empty() && self.type_params.is_empty() { return String::new() }
        let mut res = "<".to_owned();

        for (i, life) in self.lifetimes.iter().enumerate() {
            if i > 0 {
                res.push_str(", ");
            }
            res.push_str(&format!("{}", *life));
        }

        if !self.type_params.is_empty() {
            if !self.lifetimes.is_empty() {
                res.push_str(", ");
            }
            for (i, tp) in self.type_params.iter().enumerate() {
                if i > 0 {
                    res.push_str(", ");
                }
                res.push_str(&tp.name);

                if !tp.bounds.is_empty() {
                    res.push_str(&format!(": {}", TyParamBounds(&tp.bounds)));
                }

                match tp.default {
                    Some(ref ty) => { res.push_str(&format!(" = {}", ty.get_display())); },
                    None => {}
                };
            }
        }
        res.push_str(">");
        res
    }
}

/*impl<'a> fmt::Display for WhereClause<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let &WhereClause(gens) = self;
        if gens.where_predicates.is_empty() {
            return Ok(());
        }
        try!(f.write_str(&format!(" {}where{} ", WHITE, NORMAL)));
        for (i, pred) in gens.where_predicates.iter().enumerate() {
            if i > 0 {
                try!(f.write_str(", "));
            }
            match pred {
                &clean::WherePredicate::BoundPredicate { ref ty, ref bounds } => {
                    let bounds = bounds;
                    try!(write!(f, "{}: {}", ty, TyParamBounds(bounds)));
                }
                &clean::WherePredicate::RegionPredicate { ref lifetime,
                                                          ref bounds } => {
                    try!(write!(f, "{}: ", lifetime));
                    for (i, lifetime) in bounds.iter().enumerate() {
                        if i > 0 {
                            try!(f.write_str(" + "));
                        }

                        try!(write!(f, "{}", lifetime));
                    }
                }
                &clean::WherePredicate::EqPredicate { ref lhs, ref rhs } => {
                    try!(write!(f, "{} == {}", lhs, rhs));
                }
            }
        }
        Ok(())
    }
}*/

impl<'a> AsciiDisplay for WhereClause<'a> {
    fn get_display(&self) -> String {
        let &WhereClause(gens) = self;
        let mut res = String::new();
        if gens.where_predicates.is_empty() {
            return String::new();
        }
        //try!(f.write_str(&format!(" {}where{} ", WHITE, NORMAL)));
        res.push_str(&format!(" {}where{} ", WHITE, NORMAL));
        for (i, pred) in gens.where_predicates.iter().enumerate() {
            if i > 0 {
                //try!(f.write_str(", "));
                res.push_str(", ");
            }
            match pred {
                &clean::WherePredicate::BoundPredicate { ref ty, ref bounds } => {
                    let bounds = bounds;
                    //try!(write!(f, "{}: {}", ty, TyParamBounds(bounds)));
                    res.push_str(&format!("{}: {}", ty.get_display(),
                        TyParamBounds(bounds)));
                }
                &clean::WherePredicate::RegionPredicate { ref lifetime,
                                                          ref bounds } => {
                    //try!(write!(f, "{}: ", lifetime));
                    res.push_str(&format!("{}: ", lifetime));
                    for (i, lifetime) in bounds.iter().enumerate() {
                        if i > 0 {
                            //try!(f.write_str(" + "));
                            res.push_str(" + ");
                        }

                        //try!(write!(f, "{}", lifetime));
                        res.push_str(&format!("{}", lifetime));
                    }
                }
                &clean::WherePredicate::EqPredicate { ref lhs, ref rhs } => {
                    //try!(write!(f, "{} == {}", lhs, rhs));
                    res.push_str(&format!("{} == {}", lhs.get_display(), rhs.get_display()));
                }
            }
        }
        res
    }
}

/*impl fmt::Display for clean::Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(f.write_str(self.get_ref()));
        Ok(())
    }
}*/

impl AsciiDisplay for clean::Lifetime {
    fn get_display(&self) -> String {
        self.get_ref().to_owned()
    }
}

/*impl fmt::Display for clean::PolyTrait {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.lifetimes.is_empty() {
            try!(f.write_str("for<"));
            for (i, lt) in self.lifetimes.iter().enumerate() {
                if i > 0 {
                    try!(f.write_str(", "));
                }
                try!(write!(f, "{}", lt));
            }
            try!(f.write_str("> "));
        }
        write!(f, "{}", self.trait_)
    }
}*/

impl AsciiDisplay for clean::PolyTrait {
    fn get_display(&self) -> String {
        let mut res = String::new();

        if !self.lifetimes.is_empty() {
            res.push_str("for>");
            for (i, lt) in self.lifetimes.iter().enumerate() {
                if i > 0 {
                    res.push_str(", ");
                }
                res.push_str(&format!("{}", lt));
            }
            res.push_str("> ");
        }
        res.push_str(&format!("{}", self.trait_.get_display()));
        res
    }
}

/*impl fmt::Display for clean::TyParamBound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            clean::RegionBound(ref lt) => {
                write!(f, "{}", *lt)
            }
            clean::TraitBound(ref ty, modifier) => {
                let modifier_str = match modifier {
                    ast::TraitBoundModifier::None => "",
                    ast::TraitBoundModifier::Maybe => "?",
                };
                write!(f, "{}{}", modifier_str, *ty)
            }
        }
    }
}*/

impl AsciiDisplay for clean::TyParamBound {
    fn get_display(&self) -> String {
        match *self {
            clean::RegionBound(ref lt) => {
                format!("{}", *lt)
            }
            clean::TraitBound(ref ty, modifier) => {
                let modifier_str = match modifier {
                    ast::TraitBoundModifier::None => "",
                    ast::TraitBoundModifier::Maybe => "?",
                };
                format!("{}{}", modifier_str, (*ty).get_display())
            }
        }
    }
}

/*impl fmt::Display for clean::PathParameters {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            clean::PathParameters::AngleBracketed {
                ref lifetimes, ref types, ref bindings
            } => {
                if !lifetimes.is_empty() || !types.is_empty() || !bindings.is_empty() {
                    try!(f.write_str("<"));
                    let mut comma = false;
                    for lifetime in lifetimes {
                        if comma {
                            try!(f.write_str(", "));
                        }
                        comma = true;
                        try!(write!(f, "{}", *lifetime));
                    }
                    for ty in types {
                        if comma {
                            try!(f.write_str(", "));
                        }
                        comma = true;
                        try!(write!(f, "{}", *ty));
                    }
                    for binding in bindings {
                        if comma {
                            try!(f.write_str(", "));
                        }
                        comma = true;
                        try!(write!(f, "{}", *binding));
                    }
                    try!(f.write_str(">"));
                }
            }
            clean::PathParameters::Parenthesized { ref inputs, ref output } => {
                try!(f.write_str("("));
                let mut comma = false;
                for ty in inputs {
                    if comma {
                        try!(f.write_str(", "));
                    }
                    comma = true;
                    try!(write!(f, "{}", *ty));
                }
                try!(f.write_str(")"));
                if let Some(ref ty) = *output {
                    try!(f.write_str(" -> "));
                    try!(write!(f, "{}", ty));
                }
            }
        }
        Ok(())
    }
}*/

impl AsciiDisplay for clean::PathParameters {
    fn get_display(&self) -> String {
        let mut res = String::new();

        match *self {
            clean::PathParameters::AngleBracketed {
                ref lifetimes, ref types, ref bindings
            } => {
                if !lifetimes.is_empty() || !types.is_empty() || !bindings.is_empty() {
                    res.push_str("<");
                    let mut comma = false;
                    for lifetime in lifetimes {
                        if comma {
                            res.push_str(", ");
                        }
                        comma = true;
                        res.push_str(&format!("{}", *lifetime));
                    }
                    for ty in types {
                        if comma {
                            res.push_str(", ");
                        }
                        comma = true;
                        res.push_str(&format!("{}", (*ty).get_display()));
                    }
                    for binding in bindings {
                        if comma {
                            res.push_str(", ");
                        }
                        comma = true;
                        res.push_str(&format!("{}", *binding));
                    }
                    res.push_str(">");
                }
            }
            clean::PathParameters::Parenthesized { ref inputs, ref output } => {
                res.push_str("(");
                let mut comma = false;
                for ty in inputs {
                    if comma {
                        res.push_str(", ");
                    }
                    comma = true;
                    res.push_str(&format!("{}", (*ty).get_display()));
                }
                res.push_str(")");
                if let Some(ref ty) = *output {
                    res.push_str(" -> ");
                    res.push_str(&format!("{}", (*ty).get_display()));
                }
            }
        }
        res
    }
}

/*impl fmt::Display for clean::PathSegment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(f.write_str(&self.name));
        write!(f, "{}", self.params)
    }
}*/

impl AsciiDisplay for clean::PathSegment {
    fn get_display(&self) -> String {
        let mut res = String::new();

        res.push_str(&self.name);
        res.push_str(&format!("{}", self.params.get_display()));
        res
    }
}

/*impl fmt::Display for clean::Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.global {
            try!(f.write_str("::"))
        }

        for (i, seg) in self.segments.iter().enumerate() {
            if i > 0 {
                try!(f.write_str("::"))
            }
            try!(write!(f, "{}", seg));
        }
        Ok(())
    }
}*/

impl AsciiDisplay for clean::Path {
    fn get_display(&self) -> String {
        let mut res = String::new();

        if self.global {
            //try!(f.write_str("::"))
            res.push_str("::");
        }

        for (i, seg) in self.segments.iter().enumerate() {
            if i > 0 {
                //try!(f.write_str("::"))
                res.push_str("::");
            }
            //try!(write!(f, "{}", seg));
            res.push_str(&format!("{}", seg));
        }
        res
    }
}

pub fn href(did: ast::DefId) -> Option<(String, ItemType, Vec<String>)> {
    let cache = cache();
    let loc = CURRENT_LOCATION_KEY.with(|l| l.borrow().clone());
    let &(ref fqp, shortty) = match cache.paths.get(&did) {
        Some(p) => p,
        None => return None,
    };
    let mut url = if ast_util::is_local(did) || cache.inlined.contains(&did) {
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
fn resolved_path(did: ast::DefId, path: &clean::Path,
                 print_all: bool) -> String {
    let mut w = String::new();
    let last = path.segments.last().unwrap();
    let rel_root = match &*path.segments[0].name {
        "self" => Some("./".to_string()),
        _ => None,
    };

    if print_all {
        let amt = path.segments.len() - 1;
        match rel_root {
            Some(root) => {
                let mut root = String::from_str(&root);
                for seg in &path.segments[..amt] {
                    if "super" == seg.name || "self" == seg.name {
                        //try!(write!(w, "{}::", seg.name));
                        w.push_str(&format!("{}::", seg.name));
                    } else {
                        /*root.push_str(&seg.name);
                        root.push_str("/");
                        try!(write!(w, "<a class='mod'
                                            href='{}index.html'>{}</a>::",
                                      root,
                                      seg.name));*/
                        w.push_str(&format!("{}::", seg.name));
                    }
                }
            }
            None => {
                for seg in &path.segments[..amt] {
                    //try!(write!(w, "{}::", seg.name));
                    w.push_str(&format!("{}::", seg.name));
                }
            }
        }
    }

    match href(did) {
        Some((url, shortty, fqp)) => {
            /*try!(write!(w, "<a class='{}' href='{}' title='{}'>{}</a>",
                          shortty, url, fqp.connect("::"), last.name));*/
            w.push_str(&format!("{}", last.name));
        }
        _ => /*try!(write!(w, "{}", last.name))*/w.push_str(&format!("{}", last.name)),
    }
    //try!(write!(w, "{}", last.params));
    w.push_str(&format!("{}", last.params.get_display()));
    w
}

fn primitive_link(prim: clean::PrimitiveType,
                  name: &str) -> String {
    let m = cache();
    let mut f = String::new();
    let mut needs_termination = false;
    match m.primitive_locations.get(&prim) {
        Some(&ast::LOCAL_CRATE) => {
            let len = CURRENT_LOCATION_KEY.with(|s| s.borrow().len());
            let len = if len == 0 {0} else {len - 1};
            /*try!(write!(f, "{}{}{}",
                        GREEN, prim.to_url_str(), NORMAL));*/
            f.push_str(&format!("{}{}{}",
                        GREEN, prim.to_url_str(), NORMAL));
            needs_termination = true;
        }
        Some(&cnum) => {
            let path = &m.paths[&ast::DefId {
                krate: cnum,
                node: ast::CRATE_NODE_ID,
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
                    /*try!(write!(f, "{}{}{}",
                                GREEN,
                                prim.to_url_str(),
                                NORMAL));*/
                    f.push_str(&format!("{}{}{}",
                                       GREEN,
                                       prim.to_url_str(),
                                       NORMAL));
                    needs_termination = true;
                }
                None => {}
            }
        }
        None => {}
    }
    //try!(write!(f, "{}", name));
    f.push_str(name);
    /*if needs_termination {
        try!(write!(f, "</a>"));
    }*/
    f
}

/// Helper to render type parameters
fn tybounds(typarams: &Option<Vec<clean::TyParamBound> >) -> String {
    match *typarams {
        Some(ref params) => {
            let mut res = String::new();

            for param in params {
                /*try!(write!(w, " + "));
                try!(write!(w, "{}", *param));*/
                res.push_str(&format!(" + {}", *param));
            }
            res
        }
        None => String::new()
    }
}

/*impl fmt::Display for clean::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            clean::Generic(ref name) => {
                f.write_str(name)
            }
            clean::ResolvedPath{ did, ref typarams, ref path, is_generic } => {
                // Paths like T::Output and Self::Output should be rendered with all segments
                try!(resolved_path(f, did, path, is_generic));
                tybounds(f, typarams)
            }
            clean::Infer => write!(f, "_"),
            clean::Primitive(prim) => primitive_link(f, prim, prim.to_string()),
            clean::BareFunction(ref decl) => {
                write!(f, "{}{}{}fn{}{}{}",
                       BLUE,
                       UnsafetySpace(decl.unsafety),
                       match &*decl.abi {
                           "" => " extern ".to_string(),
                           "\"Rust\"" => "".to_string(),
                           s => format!(" extern {} ", s)
                       },
                       decl.generics,
                       decl.decl,
                       NORMAL)
            }
            clean::Tuple(ref typs) => {
                primitive_link(f, clean::PrimitiveTuple,
                               &*match &**typs {
                                    [ref one] => format!("({},)", one),
                                    many => format!("({})", CommaSep(&many)),
                               })
            }
            clean::Vector(ref t) => {
                primitive_link(f, clean::Slice, &format!("[{}]", **t))
            }
            clean::FixedVector(ref t, ref s) => {
                primitive_link(f, clean::PrimitiveType::Array,
                               &format!("[{}; {}]", **t, *s))
            }
            clean::Bottom => f.write_str("!"),
            clean::RawPointer(m, ref t) => {
                primitive_link(f, clean::PrimitiveType::PrimitiveRawPointer,
                               &format!("*{}{}", RawMutableSpace(m), **t))
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
                                    &format!("&{}{}[{}]", lt, m, **bt)),
                            _ => {
                                try!(primitive_link(f, clean::Slice,
                                    &format!("&{}{}[", lt, m)));
                                try!(write!(f, "{}", **bt));
                                primitive_link(f, clean::Slice, "]")
                            }
                        }
                    }
                    _ => {
                        write!(f, "&{}{}{}", lt, m, **ty)
                    }
                }
            }
            clean::PolyTraitRef(ref bounds) => {
                for (i, bound) in bounds.iter().enumerate() {
                    if i != 0 {
                        try!(write!(f, " + "));
                    }
                    try!(write!(f, "{}", *bound));
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
                try!(write!(f, "{}::", self_type));
                let path = clean::Path::singleton(name.clone());
                try!(resolved_path(f, did, &path, false));

                // FIXME: `typarams` are not rendered, and this seems bad?
                drop(typarams);
                Ok(())
            }
            clean::QPath { ref name, ref self_type, ref trait_ } => {
                write!(f, "<{} as {}>::{}", self_type, trait_, name)
            }
            clean::Unique(..) => {
                panic!("should have been cleaned")
            }
        }
    }
}*/

impl AsciiDisplay for clean::Type {
    fn get_display(&self) -> String {
        match *self {
            clean::Generic(ref name) => {
                //f.write_str(name)
                name.to_owned()
            }
            clean::ResolvedPath{ did, ref typarams, ref path, is_generic } => {
                // Paths like T::Output and Self::Output should be rendered with all segments
                let mut res = resolved_path(did, path, is_generic);
                res.push_str(&tybounds(typarams));
                res
            }
            clean::Infer => /*write!(f, "_")*/"_".to_owned(),
            clean::Primitive(prim) => primitive_link(prim, prim.to_string()),
            clean::BareFunction(ref decl) => {
                /*write!(f, "{}{}{}fn{}{}{}",
                       BLUE,
                       UnsafetySpace(decl.unsafety),
                       match &*decl.abi {
                           "" => " extern ".to_string(),
                           "\"Rust\"" => "".to_string(),
                           s => format!(" extern {} ", s)
                       },
                       decl.generics,
                       decl.decl,
                       NORMAL)*/
                format!("{}{}{}fn{}{}{}",
                       BLUE,
                       UnsafetySpace(decl.unsafety).get_display(),
                       match &*decl.abi {
                           "" => " extern ".to_string(),
                           "\"Rust\"" => "".to_string(),
                           s => format!(" extern {} ", s)
                       },
                       decl.generics.get_display(),
                       decl.decl,
                       NORMAL)
            }
            clean::Tuple(ref typs) => {
                primitive_link(clean::PrimitiveTuple,
                               &*match &**typs {
                                    [ref one] => format!("({},)", one.get_display()),
                                    many => format!("({})", CommaSep(&many)),
                               })
            }
            clean::Vector(ref t) => {
                primitive_link(clean::Slice, &format!("[{}]", (**t).get_display()))
            }
            clean::FixedVector(ref t, ref s) => {
                primitive_link(clean::PrimitiveType::Array,
                               &format!("[{}; {}]", (**t).get_display(), *s))
            }
            clean::Bottom => /*f.write_str("!")*/"!".to_owned(),
            clean::RawPointer(m, ref t) => {
                primitive_link(clean::PrimitiveType::PrimitiveRawPointer,
                               &format!("*{}{}", RawMutableSpace(m).get_display(), (**t).get_display()))
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
                                primitive_link(clean::Slice,
                                    &format!("&{}{}[{}]", lt, m.get_display(), (**bt).get_display())),
                            _ => {
                                let mut res = primitive_link(clean::Slice,
                                    &format!("&{}{}[", lt, m.get_display()));
                                //try!(write!(f, "{}", **bt));
                                res.push_str(&format!("{}", (**bt).get_display()));
                                res.push_str(&primitive_link(clean::Slice, "]"));
                                res
                            }
                        }
                    }
                    _ => {
                        /*write!(f, */format!("&{}{}{}", lt, m.get_display(), (**ty).get_display())
                    }
                }
            }
            clean::PolyTraitRef(ref bounds) => {
                let mut res = String::new();

                for (i, bound) in bounds.iter().enumerate() {
                    if i != 0 {
                        //try!(write!(f, " + "));
                        res.push_str(" + ");
                    }
                    //try!(write!(f, "{}", *bound));
                    res.push_str(&format!("{}", *bound));
                }
                res
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
                //try!(write!(f, "{}::", self_type));
                let mut res = format!("{}::", self_type.get_display());

                let path = clean::Path::singleton(name.clone());
                res.push_str(&resolved_path(did, &path, false));

                // FIXME: `typarams` are not rendered, and this seems bad?
                drop(typarams);
                res
            }
            clean::QPath { ref name, ref self_type, ref trait_ } => {
                //write!(f, "<{} as {}>::{}", self_type, trait_, name)
                format!("<{} as {}>::{}", self_type.get_display(), trait_.get_display(), name)
            }
            clean::Unique(..) => {
                panic!("should have been cleaned")
            }
        }
    }
}

/*impl<'a> fmt::Display for Method<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Method(selfty, d) = *self;
        let mut args = String::new();
        match *selfty {
            clean::SelfStatic => {},
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
        for (i, input) in d.inputs.values.iter().enumerate() {
            if i > 0 || !args.is_empty() { args.push_str(", "); }
            if !input.name.is_empty() {
                args.push_str(&format!("{}: ", input.name));
            }
            args.push_str(&format!("{}", input.type_));
        }
        write!(f, "({args}){arrow}", args = args, arrow = d.output)
    }
}*/

impl<'a> AsciiDisplay for Method<'a> {
    fn get_display(&self) -> String {
        let Method(selfty, d) = *self;
        let mut args = String::new();
        match *selfty {
            clean::SelfStatic => {},
            clean::SelfValue => args.push_str("self"),
            clean::SelfBorrowed(Some(ref lt), mtbl) => {
                args.push_str(&format!("&amp;{} {}self", *lt, MutableSpace(mtbl).get_display()));
            }
            clean::SelfBorrowed(None, mtbl) => {
                args.push_str(&format!("&amp;{}self", MutableSpace(mtbl).get_display()));
            }
            clean::SelfExplicit(ref typ) => {
                args.push_str(&format!("self: {}", (*typ).get_display()));
            }
        }
        for (i, input) in d.inputs.values.iter().enumerate() {
            if i > 0 || !args.is_empty() { args.push_str(", "); }
            if !input.name.is_empty() {
                args.push_str(&format!("{}: ", input.name));
            }
            args.push_str(&format!("{}", input.type_.get_display()));
        }
        /*write!(f, */format!("({args}){arrow}", args = args, arrow = d.output.get_display())
    }
}

/*impl fmt::Display for clean::Arguments {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, input) in self.values.iter().enumerate() {
            if i > 0 { try!(write!(f, ", ")); }
            if !input.name.is_empty() {
                try!(write!(f, "{}: ", input.name));
            }
            try!(write!(f, "{}", input.type_));
        }
        Ok(())
    }
}*/

impl AsciiDisplay for clean::Arguments {
    fn get_display(&self) -> String {
        let mut res = String::new();

        for (i, input) in self.values.iter().enumerate() {
            if i > 0 {
                //try!(write!(f, ", "));
                res.push_str(", ");
            }
            if !input.name.is_empty() {
                //try!(write!(f, "{}: ", input.name));
                res.push_str(&format!("{}: ", input.name));
            }
            //try!(write!(f, "{}", input.type_));
            res.push_str(&format!("{} ", input.type_.get_display()));
        }
        res
    }
}

/*impl fmt::Display for MutableSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            MutableSpace(clean::Immutable) => Ok(()),
            MutableSpace(clean::Mutable) => write!(f, "mut "),
        }
    }
}*/

impl AsciiDisplay for MutableSpace {
    fn get_display(&self) -> String {
        match *self {
            MutableSpace(clean::Immutable) => String::new(),
            MutableSpace(clean::Mutable) => /*write!(f, "mut ")*/"mut ".to_owned(),
        }
    }
}

/*impl fmt::Display for clean::FunctionRetTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            clean::Return(clean::Tuple(ref tys)) if tys.is_empty() => Ok(()),
            clean::Return(ref ty) => write!(f, " -> {}", ty),
            clean::DefaultReturn => Ok(()),
            clean::NoReturn => write!(f, " -> !")
        }
    }
}*/

impl AsciiDisplay for clean::FunctionRetTy {
    fn get_display(&self) -> String {
        match *self {
            clean::Return(clean::Tuple(ref tys)) if tys.is_empty() => String::new(),
            clean::Return(ref ty) => /*write!(f, */format!(" -> {}", ty.get_display()),
            clean::DefaultReturn => String::new(),
            clean::NoReturn => /*write!(f, " -> !")*/" - > !".to_owned()
        }
    }
}

/*impl fmt::Display for clean::FnDecl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({args}){arrow}", args = self.inputs, arrow = self.output)
    }
}*/

impl AsciiDisplay for clean::FnDecl {
    fn get_display(&self) -> String {
        //write!(f, "({args}){arrow}", args = self.inputs, arrow = self.output)
        format!("({args}){arrow}", args = self.inputs, arrow = self.output.get_display())
    }
}

/*impl fmt::Display for VisSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.get() {
            Some(ast::Public) => write!(f, "pub "),
            Some(ast::Inherited) | None => Ok(())
        }
    }
}*/

impl AsciiDisplay for VisSpace {
    fn get_display(&self) -> String {
        match self.get() {
            Some(ast::Public) => /*write!(f, "pub ")*/"pub ".to_owned(),
            Some(ast::Inherited) | None => String::new()
        }
    }
}

/*impl fmt::Display for UnsafetySpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.get() {
            ast::Unsafety::Unsafe => write!(f, "unsafe "),
            ast::Unsafety::Normal => Ok(())
        }
    }
}*/

impl AsciiDisplay for UnsafetySpace {
    fn get_display(&self) -> String {
        match self.get() {
            ast::Unsafety::Unsafe => /*write!(f, "unsafe ")*/"unsafe ".to_owned(),
            ast::Unsafety::Normal => String::new()
        }
    }
}

/*impl fmt::Display for ConstnessSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.get() {
            ast::Constness::Const => write!(f, "const "),
            ast::Constness::NotConst => Ok(())
        }
    }
}*/

impl AsciiDisplay for ConstnessSpace {
    fn get_display(&self) -> String {
        match self.get() {
            ast::Constness::Const => /*write!(f, "const ")*/"const ".to_owned(),
            ast::Constness::NotConst => String::new()
        }
    }
}

/*impl fmt::Display for clean::Import {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            clean::SimpleImport(ref name, ref src) => {
                if *name == src.path.segments.last().unwrap().name {
                    write!(f, "use {};", *src)
                } else {
                    write!(f, "use {} as {};", *src, *name)
                }
            }
            clean::GlobImport(ref src) => {
                write!(f, "use {}::*;", *src)
            }
            clean::ImportList(ref src, ref names) => {
                try!(write!(f, "use {}::{{", *src));
                for (i, n) in names.iter().enumerate() {
                    if i > 0 {
                        try!(write!(f, ", "));
                    }
                    try!(write!(f, "{}", *n));
                }
                write!(f, "}};")
            }
        }
    }
}*/

impl AsciiDisplay for clean::Import {
    fn get_display(&self) -> String {
        match *self {
            clean::SimpleImport(ref name, ref src) => {
                if *name == src.path.segments.last().unwrap().name {
                    //write!(f, "use {};", *src)
                    format!("use {}", *src)
                } else {
                    //write!(f, "use {} as {};", *src, *name)
                    format!("use {} as {};", *src, *name)
                }
            }
            clean::GlobImport(ref src) => {
                //write!(f, "use {}::*;", *src)
                format!("use {}::*;", *src)
            }
            clean::ImportList(ref src, ref names) => {
                //try!(write!(f, "use {}::{{", *src));
                let mut res = format!("use {}::{{", *src);

                for (i, n) in names.iter().enumerate() {
                    if i > 0 {
                        //try!(write!(f, ", "));
                        res.push_str(", ");
                    }
                    //try!(write!(f, "{}", *n));
                    res.push_str(&format!("{}", *n));
                }
                //write!(f, "}};")
                res.push_str("};");
                res
            }
        }
    }
}

/*impl fmt::Display for clean::ImportSource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.did {
            Some(did) => resolved_path(f, did, &self.path, true),
            _ => {
                for (i, seg) in self.path.segments.iter().enumerate() {
                    if i > 0 {
                        try!(write!(f, "::"))
                    }
                    try!(write!(f, "{}", seg.name));
                }
                Ok(())
            }
        }
    }
}*/

impl AsciiDisplay for clean::ImportSource {
    fn get_display(&self) -> String {
        match self.did {
            Some(did) => resolved_path(did, &self.path, true),
            _ => {
                let mut res = String::new();

                for (i, seg) in self.path.segments.iter().enumerate() {
                    if i > 0 {
                        //try!(write!(f, "::"))
                        res.push_str("::");
                    }
                    //try!(write!(f, "{}", seg.name));
                    res.push_str(&format!("{}", seg.name));
                }
                res
            }
        }
    }
}

/*impl fmt::Display for clean::ViewListIdent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.source {
            Some(did) => {
                let path = clean::Path::singleton(self.name.clone());
                resolved_path(f, did, &path, false)
            }
            _ => write!(f, "{}", self.name),
        }
    }
}*/

impl AsciiDisplay for clean::ViewListIdent {
    fn get_display(&self) -> String {
        match self.source {
            Some(did) => {
                let path = clean::Path::singleton(self.name.clone());
                resolved_path(did, &path, false)
            }
            _ => /*write!(f, */format!("{}", self.name),
        }
    }
}

/*impl fmt::Display for clean::TypeBinding {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}={}", self.name, self.ty)
    }
}*/

impl AsciiDisplay for clean::TypeBinding {
    fn get_display(&self) -> String {
        /*write!(f, */format!("{}={}", self.name, self.ty.get_display())
    }
}

/*impl fmt::Display for RawMutableSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            RawMutableSpace(clean::Immutable) => write!(f, "const "),
            RawMutableSpace(clean::Mutable) => write!(f, "mut "),
        }
    }
}*/

impl AsciiDisplay for RawMutableSpace {
    fn get_display(&self) -> String {
        match *self {
            RawMutableSpace(clean::Immutable) => /*write!(f, "const ")*/"const ".to_owned(),
            RawMutableSpace(clean::Mutable) => /*write!(f, "mut ")*/"mut ".to_owned(),
        }
    }
}

/*impl fmt::Display for AbiSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.0 {
            Abi::Rust => Ok(()),
            Abi::C => write!(f, "extern "),
            abi => write!(f, "extern {} ", abi),
        }
    }
}*/

impl AsciiDisplay for AbiSpace {
    fn get_display(&self) -> String {
        match self.0 {
            Abi::Rust => String::new(),
            Abi::C => /*write!(f, "extern ")*/"extern ".to_owned(),
            abi => /*write!(f, */format!("extern {} ", abi),
        }
    }
}