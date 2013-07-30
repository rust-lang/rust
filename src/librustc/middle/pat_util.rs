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
use syntax::codemap::span;

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
                    it: &fn(binding_mode, NodeId, span, &Path)) {
    for walk_pat(pat) |p| {
        match p.node {
          pat_ident(binding_mode, ref pth, _) if pat_is_binding(dm, p) => {
            it(binding_mode, p.id, p.span, pth);
          }
          _ => {}
        }
    }
}

pub fn pat_binding_ids(dm: resolve::DefMap, pat: @pat) -> ~[NodeId] {
    let mut found = ~[];
    pat_bindings(dm, pat, |_bm, b_id, _sp, _pt| found.push(b_id) );
    return found;
}
