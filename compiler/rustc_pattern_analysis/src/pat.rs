//! As explained in [`crate::usefulness`], values and patterns are made from constructors applied to
//! fields. This file defines types that represent patterns in this way.
use std::cell::Cell;
use std::fmt;

use smallvec::{smallvec, SmallVec};

use rustc_data_structures::captures::Captures;
use rustc_middle::ty::{self, Ty};
use rustc_span::{Span, DUMMY_SP};

use self::Constructor::*;
use self::SliceKind::*;

use crate::constructor::{Constructor, SliceKind};
use crate::cx::MatchCheckCtxt;
use crate::usefulness::PatCtxt;

/// Values and patterns can be represented as a constructor applied to some fields. This represents
/// a pattern in this form.
/// This also uses interior mutability to keep track of whether the pattern has been found reachable
/// during analysis. For this reason they cannot be cloned.
/// A `DeconstructedPat` will almost always come from user input; the only exception are some
/// `Wildcard`s introduced during specialization.
///
/// Note that the number of fields may not match the fields declared in the original struct/variant.
/// This happens if a private or `non_exhaustive` field is uninhabited, because the code mustn't
/// observe that it is uninhabited. In that case that field is not included in `fields`. Care must
/// be taken when converting to/from `thir::Pat`.
pub struct DeconstructedPat<'p, 'tcx> {
    ctor: Constructor<'tcx>,
    fields: &'p [DeconstructedPat<'p, 'tcx>],
    ty: Ty<'tcx>,
    span: Span,
    /// Whether removing this arm would change the behavior of the match expression.
    useful: Cell<bool>,
}

impl<'p, 'tcx> DeconstructedPat<'p, 'tcx> {
    pub(super) fn wildcard(ty: Ty<'tcx>, span: Span) -> Self {
        Self::new(Wildcard, &[], ty, span)
    }

    pub(super) fn new(
        ctor: Constructor<'tcx>,
        fields: &'p [DeconstructedPat<'p, 'tcx>],
        ty: Ty<'tcx>,
        span: Span,
    ) -> Self {
        DeconstructedPat { ctor, fields, ty, span, useful: Cell::new(false) }
    }

    pub(super) fn is_or_pat(&self) -> bool {
        matches!(self.ctor, Or)
    }
    /// Expand this (possibly-nested) or-pattern into its alternatives.
    pub(super) fn flatten_or_pat(&'p self) -> SmallVec<[&'p Self; 1]> {
        if self.is_or_pat() {
            self.iter_fields().flat_map(|p| p.flatten_or_pat()).collect()
        } else {
            smallvec![self]
        }
    }

    pub fn ctor(&self) -> &Constructor<'tcx> {
        &self.ctor
    }
    pub fn ty(&self) -> Ty<'tcx> {
        self.ty
    }
    pub fn span(&self) -> Span {
        self.span
    }

    pub fn iter_fields<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'p DeconstructedPat<'p, 'tcx>> + Captures<'a> {
        self.fields.iter()
    }

    /// Specialize this pattern with a constructor.
    /// `other_ctor` can be different from `self.ctor`, but must be covered by it.
    pub(super) fn specialize<'a>(
        &'a self,
        pcx: &PatCtxt<'_, 'p, 'tcx>,
        other_ctor: &Constructor<'tcx>,
    ) -> SmallVec<[&'p DeconstructedPat<'p, 'tcx>; 2]> {
        match (&self.ctor, other_ctor) {
            (Wildcard, _) => {
                // We return a wildcard for each field of `other_ctor`.
                pcx.cx.ctor_wildcard_fields(other_ctor, pcx.ty).iter().collect()
            }
            (Slice(self_slice), Slice(other_slice))
                if self_slice.arity() != other_slice.arity() =>
            {
                // The only tricky case: two slices of different arity. Since `self_slice` covers
                // `other_slice`, `self_slice` must be `VarLen`, i.e. of the form
                // `[prefix, .., suffix]`. Moreover `other_slice` is guaranteed to have a larger
                // arity. So we fill the middle part with enough wildcards to reach the length of
                // the new, larger slice.
                match self_slice.kind {
                    FixedLen(_) => bug!("{:?} doesn't cover {:?}", self_slice, other_slice),
                    VarLen(prefix, suffix) => {
                        let (ty::Slice(inner_ty) | ty::Array(inner_ty, _)) = *self.ty.kind() else {
                            bug!("bad slice pattern {:?} {:?}", self.ctor, self.ty);
                        };
                        let prefix = &self.fields[..prefix];
                        let suffix = &self.fields[self_slice.arity() - suffix..];
                        let wildcard: &_ = pcx
                            .cx
                            .pattern_arena
                            .alloc(DeconstructedPat::wildcard(inner_ty, DUMMY_SP));
                        let extra_wildcards = other_slice.arity() - self_slice.arity();
                        let extra_wildcards = (0..extra_wildcards).map(|_| wildcard);
                        prefix.iter().chain(extra_wildcards).chain(suffix).collect()
                    }
                }
            }
            _ => self.fields.iter().collect(),
        }
    }

    /// We keep track for each pattern if it was ever useful during the analysis. This is used
    /// with `redundant_spans` to report redundant subpatterns arising from or patterns.
    pub(super) fn set_useful(&self) {
        self.useful.set(true)
    }
    pub(super) fn is_useful(&self) -> bool {
        if self.useful.get() {
            true
        } else if self.is_or_pat() && self.iter_fields().any(|f| f.is_useful()) {
            // We always expand or patterns in the matrix, so we will never see the actual
            // or-pattern (the one with constructor `Or`) in the column. As such, it will not be
            // marked as useful itself, only its children will. We recover this information here.
            self.set_useful();
            true
        } else {
            false
        }
    }

    /// Report the spans of subpatterns that were not useful, if any.
    pub(super) fn redundant_spans(&self) -> Vec<Span> {
        let mut spans = Vec::new();
        self.collect_redundant_spans(&mut spans);
        spans
    }
    fn collect_redundant_spans(&self, spans: &mut Vec<Span>) {
        // We don't look at subpatterns if we already reported the whole pattern as redundant.
        if !self.is_useful() {
            spans.push(self.span);
        } else {
            for p in self.iter_fields() {
                p.collect_redundant_spans(spans);
            }
        }
    }
}

/// This is mostly copied from the `Pat` impl. This is best effort and not good enough for a
/// `Display` impl.
impl<'p, 'tcx> fmt::Debug for DeconstructedPat<'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        MatchCheckCtxt::debug_pat(f, self)
    }
}

/// Same idea as `DeconstructedPat`, except this is a fictitious pattern built up for diagnostics
/// purposes. As such they don't use interning and can be cloned.
#[derive(Debug, Clone)]
pub struct WitnessPat<'tcx> {
    ctor: Constructor<'tcx>,
    pub(crate) fields: Vec<WitnessPat<'tcx>>,
    ty: Ty<'tcx>,
}

impl<'tcx> WitnessPat<'tcx> {
    pub(super) fn new(ctor: Constructor<'tcx>, fields: Vec<Self>, ty: Ty<'tcx>) -> Self {
        Self { ctor, fields, ty }
    }
    pub(super) fn wildcard(ty: Ty<'tcx>) -> Self {
        Self::new(Wildcard, Vec::new(), ty)
    }

    /// Construct a pattern that matches everything that starts with this constructor.
    /// For example, if `ctor` is a `Constructor::Variant` for `Option::Some`, we get the pattern
    /// `Some(_)`.
    pub(super) fn wild_from_ctor(pcx: &PatCtxt<'_, '_, 'tcx>, ctor: Constructor<'tcx>) -> Self {
        let field_tys =
            pcx.cx.ctor_wildcard_fields(&ctor, pcx.ty).iter().map(|deco_pat| deco_pat.ty());
        let fields = field_tys.map(|ty| Self::wildcard(ty)).collect();
        Self::new(ctor, fields, pcx.ty)
    }

    pub fn ctor(&self) -> &Constructor<'tcx> {
        &self.ctor
    }
    pub fn ty(&self) -> Ty<'tcx> {
        self.ty
    }

    pub fn iter_fields<'a>(&'a self) -> impl Iterator<Item = &'a WitnessPat<'tcx>> {
        self.fields.iter()
    }
}
