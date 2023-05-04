use rustc_hir::def_id::DefId;

use crate::ty::{Predicate, TraitRef, Ty};

/// The "header" of an impl is everything outside the body: a Self type, a trait
/// ref (in the case of a trait impl), and a set of predicates (from the
/// bounds / where-clauses).
#[derive(Clone, Debug, TypeFoldable, TypeVisitable)]
pub struct ImplHeader<'tcx> {
    pub impl_def_id: DefId,
    pub self_ty: Ty<'tcx>,
    pub trait_ref: Option<TraitRef<'tcx>>,
    pub predicates: Vec<Predicate<'tcx>>,
}
