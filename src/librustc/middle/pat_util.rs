// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::def::*;
use middle::def_id::DefId;
use middle::ty;
use util::nodemap::FnvHashMap;

use syntax::ast;
use rustc_front::hir;
use rustc_front::util::walk_pat;
use syntax::codemap::{respan, Span, Spanned, DUMMY_SP};

use std::cell::RefCell;

pub type PatIdMap = FnvHashMap<ast::Name, ast::NodeId>;

// This is used because same-named variables in alternative patterns need to
// use the NodeId of their namesake in the first pattern.
pub fn pat_id_map(dm: &RefCell<DefMap>, pat: &hir::Pat) -> PatIdMap {
    let mut map = FnvHashMap();
    pat_bindings(dm, pat, |_bm, p_id, _s, path1| {
        map.insert(path1.node, p_id);
    });
    map
}

pub fn pat_is_refutable(dm: &DefMap, pat: &hir::Pat) -> bool {
    match pat.node {
        hir::PatLit(_) | hir::PatRange(_, _) | hir::PatQPath(..) => true,
        hir::PatEnum(_, _) |
        hir::PatIdent(_, _, None) |
        hir::PatStruct(..) => {
            match dm.get(&pat.id).map(|d| d.full_def()) {
                Some(DefVariant(..)) => true,
                _ => false
            }
        }
        hir::PatVec(_, _, _) => true,
        _ => false
    }
}

pub fn pat_is_variant_or_struct(dm: &DefMap, pat: &hir::Pat) -> bool {
    match pat.node {
        hir::PatEnum(_, _) |
        hir::PatIdent(_, _, None) |
        hir::PatStruct(..) => {
            match dm.get(&pat.id).map(|d| d.full_def()) {
                Some(DefVariant(..)) | Some(DefStruct(..)) => true,
                _ => false
            }
        }
        _ => false
    }
}

pub fn pat_is_const(dm: &DefMap, pat: &hir::Pat) -> bool {
    match pat.node {
        hir::PatIdent(_, _, None) | hir::PatEnum(..) | hir::PatQPath(..) => {
            match dm.get(&pat.id).map(|d| d.full_def()) {
                Some(DefConst(..)) | Some(DefAssociatedConst(..)) => true,
                _ => false
            }
        }
        _ => false
    }
}

// Same as above, except that partially-resolved defs cause `false` to be
// returned instead of a panic.
pub fn pat_is_resolved_const(dm: &DefMap, pat: &hir::Pat) -> bool {
    match pat.node {
        hir::PatIdent(_, _, None) | hir::PatEnum(..) | hir::PatQPath(..) => {
            match dm.get(&pat.id)
                    .and_then(|d| if d.depth == 0 { Some(d.base_def) }
                                  else { None } ) {
                Some(DefConst(..)) | Some(DefAssociatedConst(..)) => true,
                _ => false
            }
        }
        _ => false
    }
}

pub fn pat_is_binding(dm: &DefMap, pat: &hir::Pat) -> bool {
    match pat.node {
        hir::PatIdent(..) => {
            !pat_is_variant_or_struct(dm, pat) &&
            !pat_is_const(dm, pat)
        }
        _ => false
    }
}

pub fn pat_is_binding_or_wild(dm: &DefMap, pat: &hir::Pat) -> bool {
    match pat.node {
        hir::PatIdent(..) => pat_is_binding(dm, pat),
        hir::PatWild => true,
        _ => false
    }
}

/// Call `it` on every "binding" in a pattern, e.g., on `a` in
/// `match foo() { Some(a) => (), None => () }`
pub fn pat_bindings<I>(dm: &RefCell<DefMap>, pat: &hir::Pat, mut it: I) where
    I: FnMut(hir::BindingMode, ast::NodeId, Span, &Spanned<ast::Name>),
{
    walk_pat(pat, |p| {
        match p.node {
          hir::PatIdent(binding_mode, ref pth, _) if pat_is_binding(&dm.borrow(), p) => {
            it(binding_mode, p.id, p.span, &respan(pth.span, pth.node.name));
          }
          _ => {}
        }
        true
    });
}
pub fn pat_bindings_ident<I>(dm: &RefCell<DefMap>, pat: &hir::Pat, mut it: I) where
    I: FnMut(hir::BindingMode, ast::NodeId, Span, &Spanned<hir::Ident>),
{
    walk_pat(pat, |p| {
        match p.node {
          hir::PatIdent(binding_mode, ref pth, _) if pat_is_binding(&dm.borrow(), p) => {
            it(binding_mode, p.id, p.span, &respan(pth.span, pth.node));
          }
          _ => {}
        }
        true
    });
}

/// Checks if the pattern contains any patterns that bind something to
/// an ident, e.g. `foo`, or `Foo(foo)` or `foo @ Bar(..)`.
pub fn pat_contains_bindings(dm: &DefMap, pat: &hir::Pat) -> bool {
    let mut contains_bindings = false;
    walk_pat(pat, |p| {
        if pat_is_binding(dm, p) {
            contains_bindings = true;
            false // there's at least one binding, can short circuit now.
        } else {
            true
        }
    });
    contains_bindings
}

/// Checks if the pattern contains any `ref` or `ref mut` bindings,
/// and if yes wether its containing mutable ones or just immutables ones.
pub fn pat_contains_ref_binding(dm: &RefCell<DefMap>, pat: &hir::Pat) -> Option<hir::Mutability> {
    let mut result = None;
    pat_bindings(dm, pat, |mode, _, _, _| {
        match mode {
            hir::BindingMode::BindByRef(m) => {
                // Pick Mutable as maximum
                match result {
                    None | Some(hir::MutImmutable) => result = Some(m),
                    _ => (),
                }
            }
            hir::BindingMode::BindByValue(_) => { }
        }
    });
    result
}

/// Checks if the patterns for this arm contain any `ref` or `ref mut`
/// bindings, and if yes wether its containing mutable ones or just immutables ones.
pub fn arm_contains_ref_binding(dm: &RefCell<DefMap>, arm: &hir::Arm) -> Option<hir::Mutability> {
    arm.pats.iter()
            .filter_map(|pat| pat_contains_ref_binding(dm, pat))
            .max_by_key(|m| match *m {
                hir::MutMutable => 1,
                hir::MutImmutable => 0,
            })
}

/// Checks if the pattern contains any patterns that bind something to
/// an ident or wildcard, e.g. `foo`, or `Foo(_)`, `foo @ Bar(..)`,
pub fn pat_contains_bindings_or_wild(dm: &DefMap, pat: &hir::Pat) -> bool {
    let mut contains_bindings = false;
    walk_pat(pat, |p| {
        if pat_is_binding_or_wild(dm, p) {
            contains_bindings = true;
            false // there's at least one binding/wildcard, can short circuit now.
        } else {
            true
        }
    });
    contains_bindings
}

pub fn simple_name<'a>(pat: &'a hir::Pat) -> Option<ast::Name> {
    match pat.node {
        hir::PatIdent(hir::BindByValue(_), ref path1, None) => {
            Some(path1.node.name)
        }
        _ => {
            None
        }
    }
}

pub fn def_to_path(tcx: &ty::ctxt, id: DefId) -> hir::Path {
    tcx.with_path(id, |path| hir::Path {
        global: false,
        segments: path.last().map(|elem| hir::PathSegment {
            identifier: hir::Ident::from_name(elem.name()),
            parameters: hir::PathParameters::none(),
        }).into_iter().collect(),
        span: DUMMY_SP,
    })
}

/// Return variants that are necessary to exist for the pattern to match.
pub fn necessary_variants(dm: &DefMap, pat: &hir::Pat) -> Vec<DefId> {
    let mut variants = vec![];
    walk_pat(pat, |p| {
        match p.node {
            hir::PatEnum(_, _) |
            hir::PatIdent(_, _, None) |
            hir::PatStruct(..) => {
                match dm.get(&p.id) {
                    Some(&PathResolution { base_def: DefVariant(_, id, _), .. }) => {
                        variants.push(id);
                    }
                    _ => ()
                }
            }
            _ => ()
        }
        true
    });
    variants.sort();
    variants.dedup();
    variants
}
