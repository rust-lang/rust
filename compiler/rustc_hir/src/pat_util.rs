use crate::def::{CtorOf, DefKind, Res};
use crate::def_id::DefId;
use crate::hir::{self, BindingAnnotation, ByRef, HirId, PatKind};
use rustc_data_structures::fx::FxHashSet;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::symbol::Ident;
use rustc_span::Span;

use std::iter::{Enumerate, ExactSizeIterator};

pub struct EnumerateAndAdjust<I> {
    enumerate: Enumerate<I>,
    gap_pos: usize,
    gap_len: usize,
}

impl<I> Iterator for EnumerateAndAdjust<I>
where
    I: Iterator,
{
    type Item = (usize, <I as Iterator>::Item);

    fn next(&mut self) -> Option<(usize, <I as Iterator>::Item)> {
        self.enumerate
            .next()
            .map(|(i, elem)| (if i < self.gap_pos { i } else { i + self.gap_len }, elem))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.enumerate.size_hint()
    }
}

pub trait EnumerateAndAdjustIterator {
    fn enumerate_and_adjust(
        self,
        expected_len: usize,
        gap_pos: Option<usize>,
    ) -> EnumerateAndAdjust<Self>
    where
        Self: Sized;
}

impl<T: ExactSizeIterator> EnumerateAndAdjustIterator for T {
    fn enumerate_and_adjust(
        self,
        expected_len: usize,
        gap_pos: Option<usize>,
    ) -> EnumerateAndAdjust<Self>
    where
        Self: Sized,
    {
        let actual_len = self.len();
        EnumerateAndAdjust {
            enumerate: self.enumerate(),
            gap_pos: gap_pos.unwrap_or(expected_len),
            gap_len: expected_len - actual_len,
        }
    }
}

impl hir::Pat<'_> {
    /// Call `f` on every "binding" in a pattern, e.g., on `a` in
    /// `match foo() { Some(a) => (), None => () }`
    pub fn each_binding(&self, mut f: impl FnMut(hir::BindingAnnotation, HirId, Span, Ident)) {
        self.walk_always(|p| {
            if let PatKind::Binding(binding_mode, _, ident, _) = p.kind {
                f(binding_mode, p.hir_id, p.span, ident);
            }
        });
    }

    /// Call `f` on every "binding" in a pattern, e.g., on `a` in
    /// `match foo() { Some(a) => (), None => () }`.
    ///
    /// When encountering an or-pattern `p_0 | ... | p_n` only `p_0` will be visited.
    pub fn each_binding_or_first(
        &self,
        f: &mut impl FnMut(hir::BindingAnnotation, HirId, Span, Ident),
    ) {
        self.walk(|p| match &p.kind {
            PatKind::Or(ps) => {
                ps[0].each_binding_or_first(f);
                false
            }
            PatKind::Binding(bm, _, ident, _) => {
                f(*bm, p.hir_id, p.span, *ident);
                true
            }
            _ => true,
        })
    }

    pub fn simple_ident(&self) -> Option<Ident> {
        match self.kind {
            PatKind::Binding(BindingAnnotation(ByRef::No, _), _, ident, None) => Some(ident),
            _ => None,
        }
    }

    /// Returns variants that are necessary to exist for the pattern to match.
    pub fn necessary_variants(&self) -> Vec<DefId> {
        let mut variants = vec![];
        self.walk(|p| match &p.kind {
            PatKind::Or(_) => false,
            PatKind::Path(hir::QPath::Resolved(_, path))
            | PatKind::TupleStruct(hir::QPath::Resolved(_, path), ..)
            | PatKind::Struct(hir::QPath::Resolved(_, path), ..) => {
                if let Res::Def(DefKind::Variant | DefKind::Ctor(CtorOf::Variant, ..), id) =
                    path.res
                {
                    variants.push(id);
                }
                true
            }
            _ => true,
        });
        // We remove duplicates by inserting into a `FxHashSet` to avoid re-ordering
        // the bounds
        let mut duplicates = FxHashSet::default();
        variants.retain(|def_id| duplicates.insert(*def_id));
        variants
    }

    /// Checks if the pattern contains any `ref` or `ref mut` bindings, and if
    /// yes whether it contains mutable or just immutables ones.
    //
    // FIXME(tschottdorf): this is problematic as the HIR is being scraped, but
    // ref bindings are be implicit after #42640 (default match binding modes). See issue #44848.
    pub fn contains_explicit_ref_binding(&self) -> Option<hir::Mutability> {
        let mut result = None;
        self.each_binding(|annotation, _, _, _| match annotation {
            hir::BindingAnnotation::REF => match result {
                None | Some(hir::Mutability::Not) => result = Some(hir::Mutability::Not),
                _ => {}
            },
            hir::BindingAnnotation::REF_MUT => result = Some(hir::Mutability::Mut),
            _ => {}
        });
        result
    }

    /// If the pattern is `Some(<pat>)` from a desugared for loop, returns the inner pattern
    pub fn for_loop_some(&self) -> Option<&Self> {
        if self.span.desugaring_kind() == Some(DesugaringKind::ForLoop) {
            if let hir::PatKind::Struct(_, [pat_field], _) = self.kind {
                return Some(pat_field.pat);
            }
        }
        None
    }
}
