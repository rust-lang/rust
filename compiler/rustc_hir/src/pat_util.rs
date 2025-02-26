use std::iter::Enumerate;

use rustc_span::{Ident, Span};

use crate::def::{CtorOf, DefKind, Res};
use crate::def_id::{DefId, DefIdSet};
use crate::hir::{self, BindingMode, ByRef, HirId, PatKind};

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
        gap_pos: hir::DotDotPos,
    ) -> EnumerateAndAdjust<Self>
    where
        Self: Sized;
}

impl<T: ExactSizeIterator> EnumerateAndAdjustIterator for T {
    fn enumerate_and_adjust(
        self,
        expected_len: usize,
        gap_pos: hir::DotDotPos,
    ) -> EnumerateAndAdjust<Self>
    where
        Self: Sized,
    {
        let actual_len = self.len();
        EnumerateAndAdjust {
            enumerate: self.enumerate(),
            gap_pos: gap_pos.as_opt_usize().unwrap_or(expected_len),
            gap_len: expected_len - actual_len,
        }
    }
}

impl hir::Pat<'_> {
    /// Call `f` on every "binding" in a pattern, e.g., on `a` in
    /// `match foo() { Some(a) => (), None => () }`
    pub fn each_binding(&self, mut f: impl FnMut(hir::BindingMode, HirId, Span, Ident)) {
        self.walk_always(|p| {
            if let PatKind::Binding(binding_mode, _, ident, _) = p.kind {
                f(binding_mode, p.hir_id, p.span, ident);
            }
        });
    }

    /// Call `f` on every "binding" in a pattern, e.g., on `a` in
    /// `match foo() { Some(a) => (), None => () }`.
    ///
    /// When encountering an or-pattern `p_0 | ... | p_n` only the first non-never pattern will be
    /// visited. If they're all never patterns we visit nothing, which is ok since a never pattern
    /// cannot have bindings.
    pub fn each_binding_or_first(&self, f: &mut impl FnMut(hir::BindingMode, HirId, Span, Ident)) {
        self.walk(|p| match &p.kind {
            PatKind::Or(ps) => {
                for p in *ps {
                    if !p.is_never_pattern() {
                        p.each_binding_or_first(f);
                        break;
                    }
                }
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
            PatKind::Binding(BindingMode(ByRef::No, _), _, ident, None) => Some(ident),
            _ => None,
        }
    }

    /// Returns variants that are necessary to exist for the pattern to match.
    pub fn necessary_variants(&self) -> Vec<DefId> {
        let mut variants = vec![];
        self.walk(|p| match &p.kind {
            PatKind::Or(_) => false,
            PatKind::Expr(hir::PatExpr {
                kind: hir::PatExprKind::Path(hir::QPath::Resolved(_, path)),
                ..
            })
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
        // We remove duplicates by inserting into a hash set to avoid re-ordering
        // the bounds
        let mut duplicates = DefIdSet::default();
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
            hir::BindingMode::REF if result.is_none() => result = Some(hir::Mutability::Not),
            hir::BindingMode::REF_MUT => result = Some(hir::Mutability::Mut),
            _ => {}
        });
        result
    }
}
