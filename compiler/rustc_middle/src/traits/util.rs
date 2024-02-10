use rustc_data_structures::fx::FxHashSet;

use crate::ty::{Clause, PolyTraitRef, ToPolyTraitRef, ToPredicate, TyCtxt};

/// Given a [`PolyTraitRef`], get the [`Clause`]s implied by the trait's definition.
///
/// This only exists in `rustc_middle` because the more powerful elaborator depends on
/// `rustc_infer` for elaborating outlives bounds -- this should only be used for pretty
/// printing.
pub fn super_predicates_for_pretty_printing<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: PolyTraitRef<'tcx>,
) -> impl Iterator<Item = Clause<'tcx>> {
    let clause = trait_ref.to_predicate(tcx);
    Elaborator { tcx, visited: FxHashSet::from_iter([clause]), stack: vec![clause] }
}

/// Like [`super_predicates_for_pretty_printing`], except it only returns traits and filters out
/// all other [`Clause`]s.
pub fn supertraits_for_pretty_printing<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: PolyTraitRef<'tcx>,
) -> impl Iterator<Item = PolyTraitRef<'tcx>> {
    super_predicates_for_pretty_printing(tcx, trait_ref).filter_map(|clause| {
        clause.as_trait_clause().map(|trait_clause| trait_clause.to_poly_trait_ref())
    })
}

struct Elaborator<'tcx> {
    tcx: TyCtxt<'tcx>,
    visited: FxHashSet<Clause<'tcx>>,
    stack: Vec<Clause<'tcx>>,
}

impl<'tcx> Elaborator<'tcx> {
    fn elaborate(&mut self, trait_ref: PolyTraitRef<'tcx>) {
        let super_predicates =
            self.tcx.super_predicates_of(trait_ref.def_id()).predicates.iter().filter_map(
                |&(pred, _)| {
                    let clause = pred.subst_supertrait(self.tcx, &trait_ref);
                    self.visited.insert(clause).then_some(clause)
                },
            );

        self.stack.extend(super_predicates);
    }
}

impl<'tcx> Iterator for Elaborator<'tcx> {
    type Item = Clause<'tcx>;

    fn next(&mut self) -> Option<Clause<'tcx>> {
        if let Some(clause) = self.stack.pop() {
            if let Some(trait_clause) = clause.as_trait_clause() {
                self.elaborate(trait_clause.to_poly_trait_ref());
            }
            Some(clause)
        } else {
            None
        }
    }
}
