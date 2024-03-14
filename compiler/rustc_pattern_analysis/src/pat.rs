//! As explained in [`crate::usefulness`], values and patterns are made from constructors applied to
//! fields. This file defines types that represent patterns in this way.
use std::fmt;

use smallvec::{smallvec, SmallVec};

use crate::constructor::{Constructor, Slice, SliceKind};
use crate::{PatCx, PrivateUninhabitedField};

use self::Constructor::*;

/// A globally unique id to distinguish patterns.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PatId(u32);
impl PatId {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static PAT_ID: AtomicU32 = AtomicU32::new(0);
        PatId(PAT_ID.fetch_add(1, Ordering::SeqCst))
    }
}

/// A pattern with an index denoting which field it corresponds to.
pub struct IndexedPat<Cx: PatCx> {
    pub idx: usize,
    pub pat: DeconstructedPat<Cx>,
}

/// Values and patterns can be represented as a constructor applied to some fields. This represents
/// a pattern in this form. A `DeconstructedPat` will almost always come from user input; the only
/// exception are some `Wildcard`s introduced during pattern lowering.
pub struct DeconstructedPat<Cx: PatCx> {
    ctor: Constructor<Cx>,
    fields: Vec<IndexedPat<Cx>>,
    /// The number of fields in this pattern. E.g. if the pattern is `SomeStruct { field12: true, ..
    /// }` this would be the total number of fields of the struct.
    /// This is also the same as `self.ctor.arity(self.ty)`.
    arity: usize,
    ty: Cx::Ty,
    /// Extra data to store in a pattern.
    data: Cx::PatData,
    /// Globally-unique id used to track usefulness at the level of subpatterns.
    pub(crate) uid: PatId,
}

impl<Cx: PatCx> DeconstructedPat<Cx> {
    pub fn new(
        ctor: Constructor<Cx>,
        fields: Vec<IndexedPat<Cx>>,
        arity: usize,
        ty: Cx::Ty,
        data: Cx::PatData,
    ) -> Self {
        DeconstructedPat { ctor, fields, arity, ty, data, uid: PatId::new() }
    }

    pub fn at_index(self, idx: usize) -> IndexedPat<Cx> {
        IndexedPat { idx, pat: self }
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
    /// Returns the extra data stored in a pattern.
    pub fn data(&self) -> &Cx::PatData {
        &self.data
    }
    pub fn arity(&self) -> usize {
        self.arity
    }

    pub fn iter_fields<'a>(&'a self) -> impl Iterator<Item = &'a IndexedPat<Cx>> {
        self.fields.iter()
    }

    /// Specialize this pattern with a constructor.
    /// `other_ctor` can be different from `self.ctor`, but must be covered by it.
    pub(crate) fn specialize<'a>(
        &'a self,
        other_ctor: &Constructor<Cx>,
        other_ctor_arity: usize,
    ) -> SmallVec<[PatOrWild<'a, Cx>; 2]> {
        if matches!(other_ctor, PrivateUninhabited) {
            // Skip this column.
            return smallvec![];
        }

        // Start with a slice of wildcards of the appropriate length.
        let mut fields: SmallVec<[_; 2]> = (0..other_ctor_arity).map(|_| PatOrWild::Wild).collect();
        // Fill `fields` with our fields. The arities are known to be compatible.
        match self.ctor {
            // The only non-trivial case: two slices of different arity. `other_ctor` is guaranteed
            // to have a larger arity, so we adjust the indices of the patterns in the suffix so
            // that they are correctly positioned in the larger slice.
            Slice(Slice { kind: SliceKind::VarLen(prefix, _), .. })
                if self.arity != other_ctor_arity =>
            {
                for ipat in &self.fields {
                    let new_idx = if ipat.idx < prefix {
                        ipat.idx
                    } else {
                        // Adjust the indices in the suffix.
                        ipat.idx + other_ctor_arity - self.arity
                    };
                    fields[new_idx] = PatOrWild::Pat(&ipat.pat);
                }
            }
            _ => {
                for ipat in &self.fields {
                    fields[ipat.idx] = PatOrWild::Pat(&ipat.pat);
                }
            }
        }
        fields
    }

    /// Walk top-down and call `it` in each place where a pattern occurs
    /// starting with the root pattern `walk` is called on. If `it` returns
    /// false then we will descend no further but siblings will be processed.
    pub fn walk<'a>(&'a self, it: &mut impl FnMut(&'a Self) -> bool) {
        if !it(self) {
            return;
        }

        for p in self.iter_fields() {
            p.pat.walk(it)
        }
    }
}

/// This is best effort and not good enough for a `Display` impl.
impl<Cx: PatCx> fmt::Debug for DeconstructedPat<Cx> {
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

        let mut fields: Vec<_> = (0..self.arity).map(|_| PatOrWild::Wild).collect();
        for ipat in self.iter_fields() {
            fields[ipat.idx] = PatOrWild::Pat(&ipat.pat);
        }

        match pat.ctor() {
            Struct | Variant(_) | UnionField => {
                Cx::write_variant_name(f, pat)?;
                // Without `cx`, we can't know which field corresponds to which, so we can't
                // get the names of the fields. Instead we just display everything as a tuple
                // struct, which should be good enough.
                write!(f, "(")?;
                for p in fields {
                    write!(f, "{}", start_or_comma())?;
                    write!(f, "{p:?}")?;
                }
                write!(f, ")")
            }
            // Note: given the expansion of `&str` patterns done in `expand_pattern`, we should
            // be careful to detect strings here. However a string literal pattern will never
            // be reported as a non-exhaustiveness witness, so we can ignore this issue.
            Ref => {
                write!(f, "&{:?}", &fields[0])
            }
            Slice(slice) => {
                write!(f, "[")?;
                match slice.kind {
                    SliceKind::FixedLen(_) => {
                        for p in fields {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                    }
                    SliceKind::VarLen(prefix_len, _) => {
                        for p in &fields[..prefix_len] {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                        write!(f, "{}", start_or_comma())?;
                        write!(f, "..")?;
                        for p in &fields[prefix_len..] {
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
                for pat in fields {
                    write!(f, "{}{:?}", start_or_continue(" | "), pat)?;
                }
                Ok(())
            }
            Wildcard | Missing | NonExhaustive | Hidden | PrivateUninhabited => {
                write!(f, "_ : {:?}", pat.ty())
            }
        }
    }
}

/// Represents either a pattern obtained from user input or a wildcard constructed during the
/// algorithm. Do not use `Wild` to represent a wildcard pattern comping from user input.
///
/// This is morally `Option<&'p DeconstructedPat>` where `None` is interpreted as a wildcard.
pub(crate) enum PatOrWild<'p, Cx: PatCx> {
    /// A non-user-provided wildcard, created during specialization.
    Wild,
    /// A user-provided pattern.
    Pat(&'p DeconstructedPat<Cx>),
}

impl<'p, Cx: PatCx> Clone for PatOrWild<'p, Cx> {
    fn clone(&self) -> Self {
        match self {
            PatOrWild::Wild => PatOrWild::Wild,
            PatOrWild::Pat(pat) => PatOrWild::Pat(pat),
        }
    }
}

impl<'p, Cx: PatCx> Copy for PatOrWild<'p, Cx> {}

impl<'p, Cx: PatCx> PatOrWild<'p, Cx> {
    pub(crate) fn as_pat(&self) -> Option<&'p DeconstructedPat<Cx>> {
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
            PatOrWild::Pat(pat) if pat.is_or_pat() => pat
                .iter_fields()
                .flat_map(|ipat| PatOrWild::Pat(&ipat.pat).flatten_or_pat())
                .collect(),
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
}

impl<'p, Cx: PatCx> fmt::Debug for PatOrWild<'p, Cx> {
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
pub struct WitnessPat<Cx: PatCx> {
    ctor: Constructor<Cx>,
    pub(crate) fields: Vec<WitnessPat<Cx>>,
    ty: Cx::Ty,
}

impl<Cx: PatCx> Clone for WitnessPat<Cx> {
    fn clone(&self) -> Self {
        Self { ctor: self.ctor.clone(), fields: self.fields.clone(), ty: self.ty.clone() }
    }
}

impl<Cx: PatCx> WitnessPat<Cx> {
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
        let fields = cx
            .ctor_sub_tys(&ctor, &ty)
            .filter(|(_, PrivateUninhabitedField(skip))| !skip)
            .map(|(ty, _)| Self::wildcard(ty))
            .collect();
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
