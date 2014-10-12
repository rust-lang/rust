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
use middle::resolve;
use middle::ty;

use std::collections::HashMap;
use syntax::ast::*;
use syntax::ast_util::{walk_pat};
use syntax::codemap::{Span, DUMMY_SP};
use syntax::owned_slice::OwnedSlice;

pub type PatIdMap = HashMap<Ident, NodeId>;

// This is used because same-named variables in alternative patterns need to
// use the NodeId of their namesake in the first pattern.
pub fn pat_id_map(dm: &resolve::DefMap, pat: &Pat) -> PatIdMap {
    let mut map = HashMap::new();
    pat_bindings(dm, pat, |_bm, p_id, _s, path1| {
        map.insert(path1.node, p_id);
    });
    map
}

pub fn pat_is_refutable(dm: &resolve::DefMap, pat: &Pat) -> bool {
    match pat.node {
        PatLit(_) | PatRange(_, _) => true,
        PatEnum(_, _) | PatIdent(_, _, None) | PatStruct(..) => {
            match dm.borrow().find(&pat.id) {
                Some(&DefVariant(..)) => true,
                _ => false
            }
        }
        PatVec(_, _, _) => true,
        _ => false
    }
}

pub fn pat_is_variant_or_struct(dm: &resolve::DefMap, pat: &Pat) -> bool {
    match pat.node {
        PatEnum(_, _) | PatIdent(_, _, None) | PatStruct(..) => {
            match dm.borrow().find(&pat.id) {
                Some(&DefVariant(..)) | Some(&DefStruct(..)) => true,
                _ => false
            }
        }
        _ => false
    }
}

pub fn pat_is_const(dm: &resolve::DefMap, pat: &Pat) -> bool {
    match pat.node {
        PatIdent(_, _, None) | PatEnum(..) => {
            match dm.borrow().find(&pat.id) {
                Some(&DefConst(..)) => true,
                _ => false
            }
        }
        _ => false
    }
}

pub fn pat_is_binding(dm: &resolve::DefMap, pat: &Pat) -> bool {
    match pat.node {
        PatIdent(..) => {
            !pat_is_variant_or_struct(dm, pat) &&
            !pat_is_const(dm, pat)
        }
        _ => false
    }
}

pub fn pat_is_binding_or_wild(dm: &resolve::DefMap, pat: &Pat) -> bool {
    match pat.node {
        PatIdent(..) => pat_is_binding(dm, pat),
        PatWild(_) => true,
        _ => false
    }
}

/// Call `it` on every "binding" in a pattern, e.g., on `a` in
/// `match foo() { Some(a) => (), None => () }`
pub fn pat_bindings(dm: &resolve::DefMap,
                    pat: &Pat,
                    it: |BindingMode, NodeId, Span, &SpannedIdent|) {
    walk_pat(pat, |p| {
        match p.node {
          PatIdent(binding_mode, ref pth, _) if pat_is_binding(dm, p) => {
            it(binding_mode, p.id, p.span, pth);
          }
          _ => {}
        }
        true
    });
}

/// Checks if the pattern contains any patterns that bind something to
/// an ident, e.g. `foo`, or `Foo(foo)` or `foo @ Bar(..)`.
pub fn pat_contains_bindings(dm: &resolve::DefMap, pat: &Pat) -> bool {
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

pub fn simple_identifier<'a>(pat: &'a Pat) -> Option<&'a Ident> {
    match pat.node {
        PatIdent(BindByValue(_), ref path1, None) => {
            Some(&path1.node)
        }
        _ => {
            None
        }
    }
}

pub fn def_to_path(tcx: &ty::ctxt, id: DefId) -> Path {
    ty::with_path(tcx, id, |mut path| Path {
        global: false,
        segments: path.last().map(|elem| PathSegment {
            identifier: Ident::new(elem.name()),
            lifetimes: vec!(),
            types: OwnedSlice::empty()
        }).into_iter().collect(),
        span: DUMMY_SP,
    })
}
