use crate::hir::def::{CtorOf, Res, DefKind};
use crate::hir::def_id::DefId;
use crate::hir::{self, HirId, PatKind};
use syntax::ast;
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.enumerate.size_hint()
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
            gap_pos: gap_pos.unwrap_or(expected_len),
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
                match path.res {
                    Res::Def(DefKind::Variant, _) => true,
                    _ => false
                }
            }
            PatKind::Slice(..) => true,
            _ => false
        }
    }

    /// Call `f` on every "binding" in a pattern, e.g., on `a` in
    /// `match foo() { Some(a) => (), None => () }`
    pub fn each_binding<F>(&self, mut f: F)
        where F: FnMut(hir::BindingAnnotation, HirId, Span, ast::Ident),
    {
        self.walk(|p| {
            if let PatKind::Binding(binding_mode, _, ident, _) = p.node {
                f(binding_mode, p.hir_id, p.span, ident);
            }
            true
        });
    }

    /// Checks if the pattern contains any patterns that bind something to
    /// an ident, e.g., `foo`, or `Foo(foo)` or `foo @ Bar(..)`.
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
    /// an ident or wildcard, e.g., `foo`, or `Foo(_)`, `foo @ Bar(..)`,
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

    pub fn simple_ident(&self) -> Option<ast::Ident> {
        match self.node {
            PatKind::Binding(hir::BindingAnnotation::Unannotated, _, ident, None) |
            PatKind::Binding(hir::BindingAnnotation::Mutable, _, ident, None) => Some(ident),
            _ => None,
        }
    }

    /// Returns variants that are necessary to exist for the pattern to match.
    pub fn necessary_variants(&self) -> Vec<DefId> {
        let mut variants = vec![];
        self.walk(|p| {
            match p.node {
                PatKind::Path(hir::QPath::Resolved(_, ref path)) |
                PatKind::TupleStruct(hir::QPath::Resolved(_, ref path), ..) |
                PatKind::Struct(hir::QPath::Resolved(_, ref path), ..) => {
                    match path.res {
                        Res::Def(DefKind::Variant, id) => variants.push(id),
                        Res::Def(DefKind::Ctor(CtorOf::Variant, ..), id) => variants.push(id),
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

    /// Checks if the pattern contains any `ref` or `ref mut` bindings, and if
    /// yes whether it contains mutable or just immutables ones.
    //
    // FIXME(tschottdorf): this is problematic as the HIR is being scraped, but
    // ref bindings are be implicit after #42640 (default match binding modes). See issue #44848.
    pub fn contains_explicit_ref_binding(&self) -> Option<hir::Mutability> {
        let mut result = None;
        self.each_binding(|annotation, _, _, _| {
            match annotation {
                hir::BindingAnnotation::Ref => {
                    match result {
                        None | Some(hir::MutImmutable) => result = Some(hir::MutImmutable),
                        _ => (),
                    }
                }
                hir::BindingAnnotation::RefMut => result = Some(hir::MutMutable),
                _ => (),
            }
        });
        result
    }
}

impl hir::Arm {
    /// Checks if the patterns for this arm contain any `ref` or `ref mut`
    /// bindings, and if yes whether its containing mutable ones or just immutables ones.
    pub fn contains_explicit_ref_binding(&self) -> Option<hir::Mutability> {
        // FIXME(tschottdorf): contains_explicit_ref_binding() must be removed
        // for #42640 (default match binding modes).
        //
        // See #44848.
        self.pats.iter()
                 .filter_map(|pat| pat.contains_explicit_ref_binding())
                 .max_by_key(|m| match *m {
                    hir::MutMutable => 1,
                    hir::MutImmutable => 0,
                 })
    }
}
