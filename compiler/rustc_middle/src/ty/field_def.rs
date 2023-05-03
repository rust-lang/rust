use std::hash::{Hash, Hasher};

use rustc_hir::def_id::DefId;
use rustc_span::symbol::{Ident, Symbol};

use crate::ty::{SubstsRef, Ty, TyCtxt, Visibility};

#[derive(Debug, HashStable, TyEncodable, TyDecodable)]
pub struct FieldDef {
    pub did: DefId,
    pub name: Symbol,
    pub vis: Visibility<DefId>,
}

impl PartialEq for FieldDef {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // There should be only one `FieldDef` for each `did`, therefore it is
        // fine to implement `PartialEq` only based on `did`.
        //
        // Below, we exhaustively destructure `self` so that if the definition
        // of `FieldDef` changes, a compile-error will be produced, reminding
        // us to revisit this assumption.

        let Self { did: lhs_did, name: _, vis: _ } = &self;

        let Self { did: rhs_did, name: _, vis: _ } = other;

        lhs_did == rhs_did
    }
}

impl Eq for FieldDef {}

impl Hash for FieldDef {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        // There should be only one `FieldDef` for each `did`, therefore it is
        // fine to implement `Hash` only based on `did`.
        //
        // Below, we exhaustively destructure `self` so that if the definition
        // of `FieldDef` changes, a compile-error will be produced, reminding
        // us to revisit this assumption.

        let Self { did, name: _, vis: _ } = &self;

        did.hash(s)
    }
}

impl<'tcx> FieldDef {
    /// Returns the type of this field. The resulting type is not normalized. The `subst` is
    /// typically obtained via the second field of [`TyKind::Adt`].
    ///
    /// [`TyKind::Adt`]: crate::ty::TyKind::Adt
    pub fn ty(&self, tcx: TyCtxt<'tcx>, subst: SubstsRef<'tcx>) -> Ty<'tcx> {
        tcx.type_of(self.did).subst(tcx, subst)
    }

    /// Computes the `Ident` of this variant by looking up the `Span`
    pub fn ident(&self, tcx: TyCtxt<'_>) -> Ident {
        Ident::new(self.name, tcx.def_ident_span(self.did).unwrap())
    }
}
