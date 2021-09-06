use rustc_data_structures::vec_map::VecMap;
use rustc_infer::infer::InferCtxt;
use rustc_middle::ty::{self, OpaqueTypeKey, Ty, TyCtxt, TypeFoldable};
use rustc_span::Span;
use rustc_trait_selection::opaque_types::InferCtxtExt;

use super::RegionInferenceContext;

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Resolve any opaque types that were encountered while borrow checking
    /// this item. This is then used to get the type in the `type_of` query.
    ///
    /// For example consider `fn f<'a>(x: &'a i32) -> impl Sized + 'a { x }`.
    /// This is lowered to give HIR something like
    ///
    /// type f<'a>::_Return<'_a> = impl Sized + '_a;
    /// fn f<'a>(x: &'a i32) -> f<'static>::_Return<'a> { x }
    ///
    /// When checking the return type record the type from the return and the
    /// type used in the return value. In this case they might be `_Return<'1>`
    /// and `&'2 i32` respectively.
    ///
    /// Once we to this method, we have completed region inference and want to
    /// call `infer_opaque_definition_from_instantiation` to get the inferred
    /// type of `_Return<'_a>`. `infer_opaque_definition_from_instantiation`
    /// compares lifetimes directly, so we need to map the inference variables
    /// back to concrete lifetimes: `'static`, `ReEarlyBound` or `ReFree`.
    ///
    /// First we map all the lifetimes in the concrete type to an equal
    /// universal region that occurs in the concrete type's substs, in this case
    /// this would result in `&'1 i32`. We only consider regions in the substs
    /// in case there is an equal region that does not. For example, this should
    /// be allowed:
    /// `fn f<'a: 'b, 'b: 'a>(x: *mut &'b i32) -> impl Sized + 'a { x }`
    ///
    /// Then we map the regions in both the type and the subst to their
    /// `external_name` giving `concrete_type = &'a i32`,
    /// `substs = ['static, 'a]`. This will then allow
    /// `infer_opaque_definition_from_instantiation` to determine that
    /// `_Return<'_a> = &'_a i32`.
    ///
    /// There's a slight complication around closures. Given
    /// `fn f<'a: 'a>() { || {} }` the closure's type is something like
    /// `f::<'a>::{{closure}}`. The region parameter from f is essentially
    /// ignored by type checking so ends up being inferred to an empty region.
    /// Calling `universal_upper_bound` for such a region gives `fr_fn_body`,
    /// which has no `external_name` in which case we use `'empty` as the
    /// region to pass to `infer_opaque_definition_from_instantiation`.
    #[instrument(level = "debug", skip(self, infcx))]
    pub(crate) fn infer_opaque_types(
        &self,
        infcx: &InferCtxt<'_, 'tcx>,
        opaque_ty_decls: VecMap<OpaqueTypeKey<'tcx>, Ty<'tcx>>,
        span: Span,
    ) -> VecMap<OpaqueTypeKey<'tcx>, Ty<'tcx>> {
        opaque_ty_decls
            .into_iter()
            .map(|(opaque_type_key, concrete_type)| {
                let substs = opaque_type_key.substs;
                debug!(?concrete_type, ?substs);

                let mut subst_regions = vec![self.universal_regions.fr_static];
                let universal_substs = infcx.tcx.fold_regions(substs, &mut false, |region, _| {
                    let vid = self.universal_regions.to_region_vid(region);
                    subst_regions.push(vid);
                    self.definitions[vid].external_name.unwrap_or_else(|| {
                        infcx
                            .tcx
                            .sess
                            .delay_span_bug(span, "opaque type with non-universal region substs");
                        infcx.tcx.lifetimes.re_static
                    })
                });

                subst_regions.sort();
                subst_regions.dedup();

                let universal_concrete_type =
                    infcx.tcx.fold_regions(concrete_type, &mut false, |region, _| match *region {
                        ty::ReVar(vid) => subst_regions
                            .iter()
                            .find(|ur_vid| self.eval_equal(vid, **ur_vid))
                            .and_then(|ur_vid| self.definitions[*ur_vid].external_name)
                            .unwrap_or(infcx.tcx.lifetimes.re_root_empty),
                        _ => region,
                    });

                debug!(?universal_concrete_type, ?universal_substs);

                let opaque_type_key =
                    OpaqueTypeKey { def_id: opaque_type_key.def_id, substs: universal_substs };
                let remapped_type = infcx.infer_opaque_definition_from_instantiation(
                    opaque_type_key,
                    universal_concrete_type,
                    span,
                );
                (opaque_type_key, remapped_type)
            })
            .collect()
    }

    /// Map the regions in the type to named regions. This is similar to what
    /// `infer_opaque_types` does, but can infer any universal region, not only
    /// ones from the substs for the opaque type. It also doesn't double check
    /// that the regions produced are in fact equal to the named region they are
    /// replaced with. This is fine because this function is only to improve the
    /// region names in error messages.
    pub(crate) fn name_regions<T>(&self, tcx: TyCtxt<'tcx>, ty: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        tcx.fold_regions(ty, &mut false, |region, _| match *region {
            ty::ReVar(vid) => {
                // Find something that we can name
                let upper_bound = self.approx_universal_upper_bound(vid);
                self.definitions[upper_bound].external_name.unwrap_or(region)
            }
            _ => region,
        })
    }
}
