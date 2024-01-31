//! As explained in [`crate::usefulness`], values and patterns are made from constructors applied to
//! fields. This file defines types that represent patterns in this way.
use std::cell::Cell;
use std::fmt;

use smallvec::{smallvec, SmallVec};

use crate::constructor::{Constructor, Slice, SliceKind};
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
    /// Extra data to store in a pattern. `None` if the pattern is a wildcard that does not
    /// correspond to a user-supplied pattern.
    data: Option<Cx::PatData>,
    /// Whether removing this arm would change the behavior of the match expression.
    useful: Cell<bool>,
}

impl<'p, Cx: TypeCx> DeconstructedPat<'p, Cx> {
    pub fn wildcard(ty: Cx::Ty) -> Self {
        DeconstructedPat { ctor: Wildcard, fields: &[], ty, data: None, useful: Cell::new(false) }
    }

    pub fn new(
        ctor: Constructor<Cx>,
        fields: &'p [DeconstructedPat<'p, Cx>],
        ty: Cx::Ty,
        data: Cx::PatData,
    ) -> Self {
        DeconstructedPat { ctor, fields, ty, data: Some(data), useful: Cell::new(false) }
    }

    pub(crate) fn is_or_pat(&self) -> bool {
        matches!(self.ctor, Or)
    }

    pub fn ctor(&self) -> &Constructor<Cx> {
        &self.ctor
    }
    pub fn ty(&self) -> &Cx::Ty {
        &self.ty
    }
    /// Returns the extra data stored in a pattern. Returns `None` if the pattern is a wildcard that
    /// does not correspond to a user-supplied pattern.
    pub fn data(&self) -> Option<&Cx::PatData> {
        self.data.as_ref()
    }

    pub fn iter_fields(&self) -> impl Iterator<Item = &'p DeconstructedPat<'p, Cx>> + Captures<'_> {
        self.fields.iter()
    }

    /// Specialize this pattern with a constructor.
    /// `other_ctor` can be different from `self.ctor`, but must be covered by it.
    pub(crate) fn specialize(
        &self,
        other_ctor: &Constructor<Cx>,
        ctor_arity: usize,
    ) -> SmallVec<[PatOrWild<'p, Cx>; 2]> {
        let wildcard_sub_tys = || (0..ctor_arity).map(|_| PatOrWild::Wild).collect();
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
                    fields[i] = PatOrWild::Pat(&self.fields[i]);
                }
                for i in 0..suffix {
                    fields[new_arity - 1 - i] =
                        PatOrWild::Pat(&self.fields[self.fields.len() - 1 - i]);
                }
                fields
            }
            _ => self.fields.iter().map(PatOrWild::Pat).collect(),
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

/// This is best effort and not good enough for a `Display` impl.
impl<'p, Cx: TypeCx> fmt::Debug for DeconstructedPat<'p, Cx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pat = self;
        let mut first = true;
        let mut start_or_continue = |s| {
            if first {
                first = false;
                ""
            } else {
                s
            }
        };
        let mut start_or_comma = || start_or_continue(", ");

        match pat.ctor() {
            Struct | Variant(_) | UnionField => {
                Cx::write_variant_name(f, pat)?;
                // Without `cx`, we can't know which field corresponds to which, so we can't
                // get the names of the fields. Instead we just display everything as a tuple
                // struct, which should be good enough.
                write!(f, "(")?;
                for p in pat.iter_fields() {
                    write!(f, "{}", start_or_comma())?;
                    write!(f, "{p:?}")?;
                }
                write!(f, ")")
            }
            // Note: given the expansion of `&str` patterns done in `expand_pattern`, we should
            // be careful to detect strings here. However a string literal pattern will never
            // be reported as a non-exhaustiveness witness, so we can ignore this issue.
            Ref => {
                let subpattern = pat.iter_fields().next().unwrap();
                write!(f, "&{:?}", subpattern)
            }
            Slice(slice) => {
                let mut subpatterns = pat.iter_fields();
                write!(f, "[")?;
                match slice.kind {
                    SliceKind::FixedLen(_) => {
                        for p in subpatterns {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                    }
                    SliceKind::VarLen(prefix_len, _) => {
                        for p in subpatterns.by_ref().take(prefix_len) {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                        write!(f, "{}", start_or_comma())?;
                        write!(f, "..")?;
                        for p in subpatterns {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                    }
                }
                write!(f, "]")
            }
            Bool(b) => write!(f, "{b}"),
            // Best-effort, will render signed ranges incorrectly
            IntRange(range) => write!(f, "{range:?}"),
            F32Range(lo, hi, end) => write!(f, "{lo}{end}{hi}"),
            F64Range(lo, hi, end) => write!(f, "{lo}{end}{hi}"),
            Str(value) => write!(f, "{value:?}"),
            Opaque(..) => write!(f, "<constant pattern>"),
            Or => {
                for pat in pat.iter_fields() {
                    write!(f, "{}{:?}", start_or_continue(" | "), pat)?;
                }
                Ok(())
            }
            Wildcard | Missing { .. } | NonExhaustive | Hidden => write!(f, "_ : {:?}", pat.ty()),
        }
    }
}

/// Represents either a pattern obtained from user input or a wildcard constructed during the
/// algorithm. Do not use `Wild` to represent a wildcard pattern comping from user input.
///
/// This is morally `Option<&'p DeconstructedPat>` where `None` is interpreted as a wildcard.
pub(crate) enum PatOrWild<'p, Cx: TypeCx> {
    /// A non-user-provided wildcard, created during specialization.
    Wild,
    /// A user-provided pattern.
    Pat(&'p DeconstructedPat<'p, Cx>),
}

impl<'p, Cx: TypeCx> Clone for PatOrWild<'p, Cx> {
    fn clone(&self) -> Self {
        match self {
            PatOrWild::Wild => PatOrWild::Wild,
            PatOrWild::Pat(pat) => PatOrWild::Pat(pat),
        }
    }
}

impl<'p, Cx: TypeCx> Copy for PatOrWild<'p, Cx> {}

impl<'p, Cx: TypeCx> PatOrWild<'p, Cx> {
    pub(crate) fn as_pat(&self) -> Option<&'p DeconstructedPat<'p, Cx>> {
        match self {
            PatOrWild::Wild => None,
            PatOrWild::Pat(pat) => Some(pat),
        }
    }
    pub(crate) fn ctor(self) -> &'p Constructor<Cx> {
        match self {
            PatOrWild::Wild => &Wildcard,
            PatOrWild::Pat(pat) => pat.ctor(),
        }
    }

    pub(crate) fn is_or_pat(&self) -> bool {
        match self {
            PatOrWild::Wild => false,
            PatOrWild::Pat(pat) => pat.is_or_pat(),
        }
    }

    /// Expand this (possibly-nested) or-pattern into its alternatives.
    pub(crate) fn flatten_or_pat(self) -> SmallVec<[Self; 1]> {
        match self {
            PatOrWild::Pat(pat) if pat.is_or_pat() => {
                pat.iter_fields().flat_map(|p| PatOrWild::Pat(p).flatten_or_pat()).collect()
            }
            _ => smallvec![self],
        }
    }

    /// Specialize this pattern with a constructor.
    /// `other_ctor` can be different from `self.ctor`, but must be covered by it.
    pub(crate) fn specialize(
        &self,
        other_ctor: &Constructor<Cx>,
        ctor_arity: usize,
    ) -> SmallVec<[PatOrWild<'p, Cx>; 2]> {
        match self {
            PatOrWild::Wild => (0..ctor_arity).map(|_| PatOrWild::Wild).collect(),
            PatOrWild::Pat(pat) => pat.specialize(other_ctor, ctor_arity),
        }
    }

    pub(crate) fn set_useful(&self) {
        if let PatOrWild::Pat(pat) = self {
            pat.set_useful()
        }
    }
}

impl<'p, Cx: TypeCx> fmt::Debug for PatOrWild<'p, Cx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatOrWild::Wild => write!(f, "_"),
            PatOrWild::Pat(pat) => pat.fmt(f),
        }
    }
}

/// Same idea as `DeconstructedPat`, except this is a fictitious pattern built up for diagnostics
/// purposes. As such they don't use interning and can be cloned.
#[derive(Debug)]
pub struct WitnessPat<Cx: TypeCx> {
    ctor: Constructor<Cx>,
    pub(crate) fields: Vec<WitnessPat<Cx>>,
    ty: Cx::Ty,
}

impl<Cx: TypeCx> Clone for WitnessPat<Cx> {
    fn clone(&self) -> Self {
        Self { ctor: self.ctor.clone(), fields: self.fields.clone(), ty: self.ty.clone() }
    }
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
    pub(crate) fn wild_from_ctor(cx: &Cx, ctor: Constructor<Cx>, ty: Cx::Ty) -> Self {
        let fields = cx.ctor_sub_tys(&ctor, &ty).map(|ty| Self::wildcard(ty)).collect();
        Self::new(ctor, fields, ty)
    }

    pub fn ctor(&self) -> &Constructor<Cx> {
        &self.ctor
    }
    pub fn ty(&self) -> &Cx::Ty {
        &self.ty
    }

    pub fn iter_fields(&self) -> impl Iterator<Item = &WitnessPat<Cx>> {
        self.fields.iter()
    }
}
