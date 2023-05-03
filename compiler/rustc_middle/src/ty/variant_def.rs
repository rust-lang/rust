use std::hash::{Hash, Hasher};

use rustc_hir::def_id::DefId;
use rustc_index::IndexVec;
use rustc_span::symbol::{Ident, Symbol};

use crate::ty::{AdtKind, CtorKind, FieldDef, FieldIdx, TyCtxt, VariantDiscr, VariantFlags};

/// Definition of a variant -- a struct's fields or an enum variant.
#[derive(Debug, HashStable, TyEncodable, TyDecodable)]
pub struct VariantDef {
    /// `DefId` that identifies the variant itself.
    /// If this variant belongs to a struct or union, then this is a copy of its `DefId`.
    pub def_id: DefId,
    /// `DefId` that identifies the variant's constructor.
    /// If this variant is a struct variant, then this is `None`.
    pub ctor: Option<(CtorKind, DefId)>,
    /// Variant or struct name.
    pub name: Symbol,
    /// Discriminant of this variant.
    pub discr: VariantDiscr,
    /// Fields of this variant.
    pub fields: IndexVec<FieldIdx, FieldDef>,
    /// Flags of the variant (e.g. is field list non-exhaustive)?
    flags: VariantFlags,
}

impl VariantDef {
    /// Creates a new `VariantDef`.
    ///
    /// `variant_did` is the `DefId` that identifies the enum variant (if this `VariantDef`
    /// represents an enum variant).
    ///
    /// `ctor_did` is the `DefId` that identifies the constructor of unit or
    /// tuple-variants/structs. If this is a `struct`-variant then this should be `None`.
    ///
    /// `parent_did` is the `DefId` of the `AdtDef` representing the enum or struct that
    /// owns this variant. It is used for checking if a struct has `#[non_exhaustive]` w/out having
    /// to go through the redirect of checking the ctor's attributes - but compiling a small crate
    /// requires loading the `AdtDef`s for all the structs in the universe (e.g., coherence for any
    /// built-in trait), and we do not want to load attributes twice.
    ///
    /// If someone speeds up attribute loading to not be a performance concern, they can
    /// remove this hack and use the constructor `DefId` everywhere.
    pub fn new(
        name: Symbol,
        variant_did: Option<DefId>,
        ctor: Option<(CtorKind, DefId)>,
        discr: VariantDiscr,
        fields: IndexVec<FieldIdx, FieldDef>,
        adt_kind: AdtKind,
        parent_did: DefId,
        recovered: bool,
        is_field_list_non_exhaustive: bool,
    ) -> Self {
        debug!(
            "VariantDef::new(name = {:?}, variant_did = {:?}, ctor = {:?}, discr = {:?},
             fields = {:?}, adt_kind = {:?}, parent_did = {:?})",
            name, variant_did, ctor, discr, fields, adt_kind, parent_did,
        );

        let mut flags = VariantFlags::NO_VARIANT_FLAGS;
        if is_field_list_non_exhaustive {
            flags |= VariantFlags::IS_FIELD_LIST_NON_EXHAUSTIVE;
        }

        if recovered {
            flags |= VariantFlags::IS_RECOVERED;
        }

        VariantDef { def_id: variant_did.unwrap_or(parent_did), ctor, name, discr, fields, flags }
    }

    /// Is this field list non-exhaustive?
    #[inline]
    pub fn is_field_list_non_exhaustive(&self) -> bool {
        self.flags.intersects(VariantFlags::IS_FIELD_LIST_NON_EXHAUSTIVE)
    }

    /// Was this variant obtained as part of recovering from a syntactic error?
    #[inline]
    pub fn is_recovered(&self) -> bool {
        self.flags.intersects(VariantFlags::IS_RECOVERED)
    }

    /// Computes the `Ident` of this variant by looking up the `Span`
    pub fn ident(&self, tcx: TyCtxt<'_>) -> Ident {
        Ident::new(self.name, tcx.def_ident_span(self.def_id).unwrap())
    }

    #[inline]
    pub fn ctor_kind(&self) -> Option<CtorKind> {
        self.ctor.map(|(kind, _)| kind)
    }

    #[inline]
    pub fn ctor_def_id(&self) -> Option<DefId> {
        self.ctor.map(|(_, def_id)| def_id)
    }

    /// Returns the one field in this variant.
    ///
    /// `panic!`s if there are no fields or multiple fields.
    #[inline]
    pub fn single_field(&self) -> &FieldDef {
        assert!(self.fields.len() == 1);

        &self.fields[FieldIdx::from_u32(0)]
    }
}

impl PartialEq for VariantDef {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // There should be only one `VariantDef` for each `def_id`, therefore
        // it is fine to implement `PartialEq` only based on `def_id`.
        //
        // Below, we exhaustively destructure `self` and `other` so that if the
        // definition of `VariantDef` changes, a compile-error will be produced,
        // reminding us to revisit this assumption.

        let Self { def_id: lhs_def_id, ctor: _, name: _, discr: _, fields: _, flags: _ } = &self;
        let Self { def_id: rhs_def_id, ctor: _, name: _, discr: _, fields: _, flags: _ } = other;
        lhs_def_id == rhs_def_id
    }
}

impl Eq for VariantDef {}

impl Hash for VariantDef {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        // There should be only one `VariantDef` for each `def_id`, therefore
        // it is fine to implement `Hash` only based on `def_id`.
        //
        // Below, we exhaustively destructure `self` so that if the definition
        // of `VariantDef` changes, a compile-error will be produced, reminding
        // us to revisit this assumption.

        let Self { def_id, ctor: _, name: _, discr: _, fields: _, flags: _ } = &self;
        def_id.hash(s)
    }
}
