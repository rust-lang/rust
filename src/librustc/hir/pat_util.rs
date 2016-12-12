// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def::Def;
use hir::def_id::DefId;
use hir::{self, PatKind};
use syntax::ast;
use syntax::codemap::Spanned;
use syntax_pos::Span;

use std::iter::{Enumerate, ExactSizeIterator};

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

impl hir::Pat {
    pub fn is_refutable(&self) -> bool {
        match self.node {
            PatKind::Lit(_) |
            PatKind::Range(..) |
            PatKind::Path(hir::QPath::Resolved(Some(..), _)) |
            PatKind::Path(hir::QPath::TypeRelative(..)) => true,

            PatKind::Path(hir::QPath::Resolved(_, ref path)) |
            PatKind::TupleStruct(hir::QPath::Resolved(_, ref path), ..) |
            PatKind::Struct(hir::QPath::Resolved(_, ref path), ..) => {
                match path.def {
                    Def::Variant(..) | Def::VariantCtor(..) => true,
                    _ => false
                }
            }
            PatKind::Slice(..) => true,
            _ => false
        }
    }

    pub fn is_const(&self) -> bool {
        match self.node {
            PatKind::Path(hir::QPath::TypeRelative(..)) => true,
            PatKind::Path(hir::QPath::Resolved(_, ref path)) => {
                match path.def {
                    Def::Const(..) | Def::AssociatedConst(..) => true,
                    _ => false
                }
            }
            _ => false
        }
    }

    /// Call `f` on every "binding" in a pattern, e.g., on `a` in
    /// `match foo() { Some(a) => (), None => () }`
    pub fn each_binding<F>(&self, mut f: F)
        where F: FnMut(hir::BindingMode, ast::NodeId, Span, &Spanned<ast::Name>),
    {
        self.walk(|p| {
            if let PatKind::Binding(binding_mode, _, ref pth, _) = p.node {
                f(binding_mode, p.id, p.span, pth);
            }
            true
        });
    }

    /// Checks if the pattern contains any patterns that bind something to
    /// an ident, e.g. `foo`, or `Foo(foo)` or `foo @ Bar(..)`.
    pub fn contains_bindings(&self) -> bool {
        let mut contains_bindings = false;
        self.walk(|p| {
            if let PatKind::Binding(..) = p.node {
                contains_bindings = true;
                false // there's at least one binding, can short circuit now.
            } else {
                true
            }
        });
        contains_bindings
    }

    /// Checks if the pattern contains any patterns that bind something to
    /// an ident or wildcard, e.g. `foo`, or `Foo(_)`, `foo @ Bar(..)`,
    pub fn contains_bindings_or_wild(&self) -> bool {
        let mut contains_bindings = false;
        self.walk(|p| {
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

    pub fn simple_name(&self) -> Option<ast::Name> {
        match self.node {
            PatKind::Binding(hir::BindByValue(..), _, ref path1, None) => {
                Some(path1.node)
            }
            _ => {
                None
            }
        }
    }

    /// Return variants that are necessary to exist for the pattern to match.
    pub fn necessary_variants(&self) -> Vec<DefId> {
        let mut variants = vec![];
        self.walk(|p| {
            match p.node {
                PatKind::Path(hir::QPath::Resolved(_, ref path)) |
                PatKind::TupleStruct(hir::QPath::Resolved(_, ref path), ..) |
                PatKind::Struct(hir::QPath::Resolved(_, ref path), ..) => {
                    match path.def {
                        Def::Variant(id) |
                        Def::VariantCtor(id, ..) => variants.push(id),
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

    /// Checks if the pattern contains any `ref` or `ref mut` bindings,
    /// and if yes whether its containing mutable ones or just immutables ones.
    pub fn contains_ref_binding(&self) -> Option<hir::Mutability> {
        let mut result = None;
        self.each_binding(|mode, _, _, _| {
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
}

impl hir::Arm {
    /// Checks if the patterns for this arm contain any `ref` or `ref mut`
    /// bindings, and if yes whether its containing mutable ones or just immutables ones.
    pub fn contains_ref_binding(&self) -> Option<hir::Mutability> {
        self.pats.iter()
                 .filter_map(|pat| pat.contains_ref_binding())
                 .max_by_key(|m| match *m {
                    hir::MutMutable => 1,
                    hir::MutImmutable => 0,
                 })
    }
}
