use rustc::hir::def_id::DefId;
use rustc::infer::InferCtxt;
use rustc::ty;
use rustc_data_structures::fx::FxHashMap;
use rustc_span::Span;

use super::RegionInferenceContext;

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Resolve any opaque types that were encountered while borrow checking
    /// this item. This is then used to get the type in the `type_of` query.
    pub(in crate::borrow_check) fn infer_opaque_types(
        &self,
        infcx: &InferCtxt<'_, 'tcx>,
        opaque_ty_decls: FxHashMap<DefId, ty::ResolvedOpaqueTy<'tcx>>,
        span: Span,
    ) -> FxHashMap<DefId, ty::ResolvedOpaqueTy<'tcx>> {
        opaque_ty_decls
            .into_iter()
            .map(|(opaque_def_id, ty::ResolvedOpaqueTy { concrete_type, substs })| {
                debug!(
                    "infer_opaque_types(concrete_type = {:?}, substs = {:?})",
                    concrete_type, substs
                );

                // Map back to "concrete" regions so that errors in
                // `infer_opaque_definition_from_instantiation` can show
                // sensible region names.
                let universal_concrete_type =
                    infcx.tcx.fold_regions(&concrete_type, &mut false, |region, _| match region {
                        &ty::ReVar(vid) => {
                            let universal_bound = self.universal_upper_bound(vid);
                            self.definitions[universal_bound]
                                .external_name
                                .filter(|_| self.eval_equal(universal_bound, vid))
                                .unwrap_or(infcx.tcx.lifetimes.re_empty)
                        }
                        concrete => concrete,
                    });
                let universal_substs =
                    infcx.tcx.fold_regions(&substs, &mut false, |region, _| match region {
                        ty::ReVar(vid) => {
                            self.definitions[*vid].external_name.unwrap_or_else(|| {
                                infcx.tcx.sess.delay_span_bug(
                                    span,
                                    "opaque type with non-universal region substs",
                                );
                                infcx.tcx.lifetimes.re_static
                            })
                        }
                        concrete => concrete,
                    });

                debug!(
                    "infer_opaque_types(universal_concrete_type = {:?}, universal_substs = {:?})",
                    universal_concrete_type, universal_substs
                );

                let remapped_type = infcx.infer_opaque_definition_from_instantiation(
                    opaque_def_id,
                    universal_substs,
                    universal_concrete_type,
                    span,
                );
                (
                    opaque_def_id,
                    ty::ResolvedOpaqueTy { concrete_type: remapped_type, substs: universal_substs },
                )
            })
            .collect()
    }
}
