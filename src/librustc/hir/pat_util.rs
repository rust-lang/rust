// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def::*;
use hir::def_id::DefId;
use hir::{self, PatKind};
use ty::TyCtxt;
use util::nodemap::FnvHashMap;
use syntax::ast;
use syntax::codemap::Spanned;
use syntax_pos::{Span, DUMMY_SP};

use std::iter::{Enumerate, ExactSizeIterator};

pub type PatIdMap = FnvHashMap<ast::Name, ast::NodeId>;

pub struct EnumerateAndAdjust<I> {
    enumerate: Enumerate<I>,
    gap_pos: usize,
    gap_len: usize,
}

impl<I> Iterator for EnumerateAndAdjust<I> where I: Iterator {
    type Item = (usize, <I as Iterator>::Item);

    fn next(&mut self) -> Option<(usize, <I as Iterator>::Item)> {
        self.enumerate.next().map(|(i, elem)| {
            (if i < self.gap_pos { i } else { i + self.gap_len }, elem)
        })
    }
}

pub trait EnumerateAndAdjustIterator {
    fn enumerate_and_adjust(self, expected_len: usize, gap_pos: Option<usize>)
        -> EnumerateAndAdjust<Self> where Self: Sized;
}

impl<T: ExactSizeIterator> EnumerateAndAdjustIterator for T {
    fn enumerate_and_adjust(self, expected_len: usize, gap_pos: Option<usize>)
            -> EnumerateAndAdjust<Self> where Self: Sized {
        let actual_len = self.len();
        EnumerateAndAdjust {
            enumerate: self.enumerate(),
            gap_pos: if let Some(gap_pos) = gap_pos { gap_pos } else { expected_len },
            gap_len: expected_len - actual_len,
        }
    }
}

pub fn pat_is_refutable(dm: &DefMap, pat: &hir::Pat) -> bool {
    match pat.node {
        PatKind::Lit(_) | PatKind::Range(_, _) | PatKind::QPath(..) => true,
        PatKind::TupleStruct(..) |
        PatKind::Path(..) |
        PatKind::Struct(..) => {
            match dm.get(&pat.id).map(|d| d.full_def()) {
                Some(Def::Variant(..)) => true,
                _ => false
            }
        }
        PatKind::Vec(_, _, _) => true,
        _ => false
    }
}

pub fn pat_is_variant_or_struct(dm: &DefMap, pat: &hir::Pat) -> bool {
    match pat.node {
        PatKind::TupleStruct(..) |
        PatKind::Path(..) |
        PatKind::Struct(..) => {
            match dm.get(&pat.id).map(|d| d.full_def()) {
                Some(Def::Variant(..)) | Some(Def::Struct(..)) | Some(Def::TyAlias(..)) => true,
                _ => false
            }
        }
        _ => false
    }
}

pub fn pat_is_const(dm: &DefMap, pat: &hir::Pat) -> bool {
    match pat.node {
        PatKind::Path(..) | PatKind::QPath(..) => {
            match dm.get(&pat.id).map(|d| d.full_def()) {
                Some(Def::Const(..)) | Some(Def::AssociatedConst(..)) => true,
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
        PatKind::Path(..) | PatKind::QPath(..) => {
            match dm.get(&pat.id)
                    .and_then(|d| if d.depth == 0 { Some(d.base_def) }
                                  else { None } ) {
                Some(Def::Const(..)) | Some(Def::AssociatedConst(..)) => true,
                _ => false
            }
        }
        _ => false
    }
}

/// Call `f` on every "binding" in a pattern, e.g., on `a` in
/// `match foo() { Some(a) => (), None => () }`
pub fn pat_bindings<F>(pat: &hir::Pat, mut f: F)
    where F: FnMut(hir::BindingMode, ast::NodeId, Span, &Spanned<ast::Name>),
{
    pat.walk(|p| {
        if let PatKind::Binding(binding_mode, ref pth, _) = p.node {
            f(binding_mode, p.id, p.span, pth);
        }
        true
    });
}

/// Checks if the pattern contains any patterns that bind something to
/// an ident, e.g. `foo`, or `Foo(foo)` or `foo @ Bar(..)`.
pub fn pat_contains_bindings(pat: &hir::Pat) -> bool {
    let mut contains_bindings = false;
    pat.walk(|p| {
        if let PatKind::Binding(..) = p.node {
            contains_bindings = true;
            false // there's at least one binding, can short circuit now.
        } else {
            true
        }
    });
    contains_bindings
}

/// Checks if the pattern contains any `ref` or `ref mut` bindings,
/// and if yes whether its containing mutable ones or just immutables ones.
pub fn pat_contains_ref_binding(pat: &hir::Pat) -> Option<hir::Mutability> {
    let mut result = None;
    pat_bindings(pat, |mode, _, _, _| {
        if let hir::BindingMode::BindByRef(m) = mode {
            // Pick Mutable as maximum
            match result {
                None | Some(hir::MutImmutable) => result = Some(m),
                _ => (),
            }
        }
    });
    result
}

/// Checks if the patterns for this arm contain any `ref` or `ref mut`
/// bindings, and if yes whether its containing mutable ones or just immutables ones.
pub fn arm_contains_ref_binding(arm: &hir::Arm) -> Option<hir::Mutability> {
    arm.pats.iter()
            .filter_map(|pat| pat_contains_ref_binding(pat))
            .max_by_key(|m| match *m {
                hir::MutMutable => 1,
                hir::MutImmutable => 0,
            })
}

/// Checks if the pattern contains any patterns that bind something to
/// an ident or wildcard, e.g. `foo`, or `Foo(_)`, `foo @ Bar(..)`,
pub fn pat_contains_bindings_or_wild(pat: &hir::Pat) -> bool {
    let mut contains_bindings = false;
    pat.walk(|p| {
        match p.node {
            PatKind::Binding(..) | PatKind::Wild => {
                contains_bindings = true;
                false // there's at least one binding/wildcard, can short circuit now.
            }
            _ => true
        }
    });
    contains_bindings
}

pub fn simple_name<'a>(pat: &'a hir::Pat) -> Option<ast::Name> {
    match pat.node {
        PatKind::Binding(hir::BindByValue(..), ref path1, None) => {
            Some(path1.node)
        }
        _ => {
            None
        }
    }
}

pub fn def_to_path<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, id: DefId) -> hir::Path {
    hir::Path::from_name(DUMMY_SP, tcx.item_name(id))
}

/// Return variants that are necessary to exist for the pattern to match.
pub fn necessary_variants(dm: &DefMap, pat: &hir::Pat) -> Vec<DefId> {
    let mut variants = vec![];
    pat.walk(|p| {
        match p.node {
            PatKind::TupleStruct(..) |
            PatKind::Path(..) |
            PatKind::Struct(..) => {
                match dm.get(&p.id) {
                    Some(&PathResolution { base_def: Def::Variant(_, id), .. }) => {
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
