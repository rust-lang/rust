// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::resolve;

use std::hashmap::HashMap;
use syntax::ast::*;
use syntax::ast_util::{path_to_ident, walk_pat};
use syntax::codemap::Span;

pub type PatIdMap = HashMap<ident, NodeId>;

// This is used because same-named variables in alternative patterns need to
// use the NodeId of their namesake in the first pattern.
pub fn pat_id_map(dm: resolve::DefMap, pat: @pat) -> PatIdMap {
    let mut map = HashMap::new();
    do pat_bindings(dm, pat) |_bm, p_id, _s, n| {
      map.insert(path_to_ident(n), p_id);
    };
    map
}

pub fn pat_is_variant_or_struct(dm: resolve::DefMap, pat: &pat) -> bool {
    match pat.node {
        pat_enum(_, _) | pat_ident(_, _, None) | pat_struct(*) => {
            match dm.find(&pat.id) {
                Some(&def_variant(*)) | Some(&def_struct(*)) => true,
                _ => false
            }
        }
        _ => false
    }
}

pub fn pat_is_const(dm: resolve::DefMap, pat: &pat) -> bool {
    match pat.node {
        pat_ident(_, _, None) | pat_enum(*) => {
            match dm.find(&pat.id) {
                Some(&def_static(_, false)) => true,
                _ => false
            }
        }
        _ => false
    }
}

pub fn pat_is_binding(dm: resolve::DefMap, pat: @pat) -> bool {
    match pat.node {
        pat_ident(*) => {
            !pat_is_variant_or_struct(dm, pat) &&
            !pat_is_const(dm, pat)
        }
        _ => false
    }
}

pub fn pat_is_binding_or_wild(dm: resolve::DefMap, pat: @pat) -> bool {
    match pat.node {
        pat_ident(*) => pat_is_binding(dm, pat),
        pat_wild => true,
        _ => false
    }
}

pub fn pat_bindings(dm: resolve::DefMap, pat: @pat,
                    it: &fn(binding_mode, NodeId, Span, &Path)) {
    do walk_pat(pat) |p| {
        match p.node {
          pat_ident(binding_mode, ref pth, _) if pat_is_binding(dm, p) => {
            it(binding_mode, p.id, p.span, pth);
          }
          _ => {}
        }
        true
    };
}

pub fn pat_binding_ids(dm: resolve::DefMap, pat: @pat) -> ~[NodeId] {
    let mut found = ~[];
    pat_bindings(dm, pat, |_bm, b_id, _sp, _pt| found.push(b_id) );
    return found;
}

/// Checks if the pattern contains any patterns that bind something to
/// an ident, e.g. `foo`, or `Foo(foo)` or `foo @ Bar(*)`.
pub fn pat_contains_bindings(dm: resolve::DefMap, pat: @pat) -> bool {
    let mut contains_bindings = false;
    do walk_pat(pat) |p| {
        if pat_is_binding(dm, p) {
            contains_bindings = true;
            false // there's at least one binding, can short circuit now.
        } else {
            true
        }
    };
    contains_bindings
}
