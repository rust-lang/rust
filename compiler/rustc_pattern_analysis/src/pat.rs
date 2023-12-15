//! As explained in [`crate::usefulness`], values and patterns are made from constructors applied to
//! fields. This file defines types that represent patterns in this way.
use std::cell::Cell;
use std::fmt;

use smallvec::{smallvec, SmallVec};

use crate::constructor::{Constructor, Slice, SliceKind};
use crate::usefulness::PlaceCtxt;
use crate::{Captures, TypeCx};

use self::Constructor::*;

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
pub struct DeconstructedPat<'p, Cx: TypeCx> {
    ctor: Constructor<Cx>,
    fields: &'p [DeconstructedPat<'p, Cx>],
    ty: Cx::Ty,
    data: Cx::PatData,
    /// Whether removing this arm would change the behavior of the match expression.
    useful: Cell<bool>,
}

impl<'p, Cx: TypeCx> DeconstructedPat<'p, Cx> {
    pub fn wildcard(ty: Cx::Ty, data: Cx::PatData) -> Self {
        Self::new(Wildcard, &[], ty, data)
    }

    pub fn new(
        ctor: Constructor<Cx>,
        fields: &'p [DeconstructedPat<'p, Cx>],
        ty: Cx::Ty,
        data: Cx::PatData,
    ) -> Self {
        DeconstructedPat { ctor, fields, ty, data, useful: Cell::new(false) }
    }

    pub(crate) fn is_or_pat(&self) -> bool {
        matches!(self.ctor, Or)
    }
    /// Expand this (possibly-nested) or-pattern into its alternatives.
    pub(crate) fn flatten_or_pat(&self) -> SmallVec<[&Self; 1]> {
        if self.is_or_pat() {
            self.iter_fields().flat_map(|p| p.flatten_or_pat()).collect()
        } else {
            smallvec![self]
        }
    }

    pub fn ctor(&self) -> &Constructor<Cx> {
        &self.ctor
    }
    pub fn ty(&self) -> Cx::Ty {
        self.ty
    }
    pub fn data(&self) -> &Cx::PatData {
        &self.data
    }

    pub fn iter_fields<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'p DeconstructedPat<'p, Cx>> + Captures<'a> {
        self.fields.iter()
    }

    /// Specialize this pattern with a constructor.
    /// `other_ctor` can be different from `self.ctor`, but must be covered by it.
    pub(crate) fn specialize<'a>(
        &self,
        pcx: &PlaceCtxt<'a, 'p, Cx>,
        other_ctor: &Constructor<Cx>,
    ) -> SmallVec<[&'a DeconstructedPat<'p, Cx>; 2]> {
        let wildcard_sub_tys = || {
            let tys = pcx.ctor_sub_tys(other_ctor);
            tys.iter()
                .map(|ty| DeconstructedPat::wildcard(*ty, Cx::PatData::default()))
                .map(|pat| pcx.mcx.wildcard_arena.alloc(pat) as &_)
                .collect()
        };
        match (&self.ctor, other_ctor) {
            // Return a wildcard for each field of `other_ctor`.
            (Wildcard, _) => wildcard_sub_tys(),
            // The only non-trivial case: two slices of different arity. `other_slice` is
            // guaranteed to have a larger arity, so we fill the middle part with enough
            // wildcards to reach the length of the new, larger slice.
            (
                &Slice(self_slice @ Slice { kind: SliceKind::VarLen(prefix, suffix), .. }),
                &Slice(other_slice),
            ) if self_slice.arity() != other_slice.arity() => {
                // Start with a slice of wildcards of the appropriate length.
                let mut fields: SmallVec<[_; 2]> = wildcard_sub_tys();
                // Fill in the fields from both ends.
                let new_arity = fields.len();
                for i in 0..prefix {
                    fields[i] = &self.fields[i];
                }
                for i in 0..suffix {
                    fields[new_arity - 1 - i] = &self.fields[self.fields.len() - 1 - i];
                }
                fields
            }
            _ => self.fields.iter().collect(),
        }
    }

    /// We keep track for each pattern if it was ever useful during the analysis. This is used with
    /// `redundant_subpatterns` to report redundant subpatterns arising from or patterns.
    pub(crate) fn set_useful(&self) {
        self.useful.set(true)
    }
    pub(crate) fn is_useful(&self) -> bool {
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

    /// Report the subpatterns that were not useful, if any.
    pub(crate) fn redundant_subpatterns(&self) -> Vec<&Self> {
        let mut subpats = Vec::new();
        self.collect_redundant_subpatterns(&mut subpats);
        subpats
    }
    fn collect_redundant_subpatterns<'a>(&'a self, subpats: &mut Vec<&'a Self>) {
        // We don't look at subpatterns if we already reported the whole pattern as redundant.
        if !self.is_useful() {
            subpats.push(self);
        } else {
            for p in self.iter_fields() {
                p.collect_redundant_subpatterns(subpats);
            }
        }
    }
}

/// This is mostly copied from the `Pat` impl. This is best effort and not good enough for a
/// `Display` impl.
impl<'p, Cx: TypeCx> fmt::Debug for DeconstructedPat<'p, Cx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Cx::debug_pat(f, self)
    }
}

/// Same idea as `DeconstructedPat`, except this is a fictitious pattern built up for diagnostics
/// purposes. As such they don't use interning and can be cloned.
#[derive(Debug, Clone)]
pub struct WitnessPat<Cx: TypeCx> {
    ctor: Constructor<Cx>,
    pub(crate) fields: Vec<WitnessPat<Cx>>,
    ty: Cx::Ty,
}

impl<Cx: TypeCx> WitnessPat<Cx> {
    pub(crate) fn new(ctor: Constructor<Cx>, fields: Vec<Self>, ty: Cx::Ty) -> Self {
        Self { ctor, fields, ty }
    }
    pub(crate) fn wildcard(ty: Cx::Ty) -> Self {
        Self::new(Wildcard, Vec::new(), ty)
    }

    /// Construct a pattern that matches everything that starts with this constructor.
    /// For example, if `ctor` is a `Constructor::Variant` for `Option::Some`, we get the pattern
    /// `Some(_)`.
    pub(crate) fn wild_from_ctor(pcx: &PlaceCtxt<'_, '_, Cx>, ctor: Constructor<Cx>) -> Self {
        let field_tys = pcx.ctor_sub_tys(&ctor);
        let fields = field_tys.iter().map(|ty| Self::wildcard(*ty)).collect();
        Self::new(ctor, fields, pcx.ty)
    }

    pub fn ctor(&self) -> &Constructor<Cx> {
        &self.ctor
    }
    pub fn ty(&self) -> Cx::Ty {
        self.ty
    }

    pub fn iter_fields<'a>(&'a self) -> impl Iterator<Item = &'a WitnessPat<Cx>> {
        self.fields.iter()
    }
}
