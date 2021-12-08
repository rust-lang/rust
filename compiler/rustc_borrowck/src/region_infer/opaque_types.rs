use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::vec_map::VecMap;
use rustc_hir::OpaqueTyOrigin;
use rustc_infer::infer::opaque_types::OpaqueTypeDecl;
use rustc_infer::infer::InferCtxt;
use rustc_middle::ty::subst::GenericArgKind;
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
        opaque_ty_decls: VecMap<OpaqueTypeKey<'tcx>, OpaqueTypeDecl<'tcx>>,
        span: Span,
    ) -> VecMap<OpaqueTypeKey<'tcx>, Ty<'tcx>> {
        opaque_ty_decls
            .into_iter()
            .filter_map(|(opaque_type_key, decl)| {
                let substs = opaque_type_key.substs;
                let concrete_type = decl.concrete_ty;
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

                check_opaque_type_parameter_valid(
                    infcx.tcx,
                    opaque_type_key,
                    OpaqueTypeDecl { concrete_ty: remapped_type, ..decl },
                )
                .then_some((opaque_type_key, remapped_type))
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
                let upper_bound = &self.definitions[upper_bound];
                match upper_bound.external_name {
                    Some(reg) => reg,
                    None => {
                        // Nothing exact found, so we pick the first one that we find.
                        let scc = self.constraint_sccs.scc(vid);
                        for vid in self.rev_scc_graph.as_ref().unwrap().upper_bounds(scc) {
                            match self.definitions[vid].external_name {
                                None => {}
                                Some(&ty::ReStatic) => {}
                                Some(region) => return region,
                            }
                        }
                        region
                    }
                }
            }
            _ => region,
        })
    }
}

fn check_opaque_type_parameter_valid(
    tcx: TyCtxt<'_>,
    opaque_type_key: OpaqueTypeKey<'_>,
    decl: OpaqueTypeDecl<'_>,
) -> bool {
    match decl.origin {
        // No need to check return position impl trait (RPIT)
        // because for type and const parameters they are correct
        // by construction: we convert
        //
        // fn foo<P0..Pn>() -> impl Trait
        //
        // into
        //
        // type Foo<P0...Pn>
        // fn foo<P0..Pn>() -> Foo<P0...Pn>.
        //
        // For lifetime parameters we convert
        //
        // fn foo<'l0..'ln>() -> impl Trait<'l0..'lm>
        //
        // into
        //
        // type foo::<'p0..'pn>::Foo<'q0..'qm>
        // fn foo<l0..'ln>() -> foo::<'static..'static>::Foo<'l0..'lm>.
        //
        // which would error here on all of the `'static` args.
        OpaqueTyOrigin::FnReturn(..) | OpaqueTyOrigin::AsyncFn(..) => return true,
        // Check these
        OpaqueTyOrigin::TyAlias => {}
    }
    let span = decl.definition_span;
    let opaque_generics = tcx.generics_of(opaque_type_key.def_id);
    let mut seen_params: FxHashMap<_, Vec<_>> = FxHashMap::default();
    for (i, arg) in opaque_type_key.substs.iter().enumerate() {
        let arg_is_param = match arg.unpack() {
            GenericArgKind::Type(ty) => matches!(ty.kind(), ty::Param(_)),
            GenericArgKind::Lifetime(ty::ReStatic) => {
                tcx.sess
                    .struct_span_err(span, "non-defining opaque type use in defining scope")
                    .span_label(
                        tcx.def_span(opaque_generics.param_at(i, tcx).def_id),
                        "cannot use static lifetime; use a bound lifetime \
                                    instead or remove the lifetime parameter from the \
                                    opaque type",
                    )
                    .emit();
                return false;
            }
            GenericArgKind::Lifetime(lt) => {
                matches!(lt, ty::ReEarlyBound(_) | ty::ReFree(_))
            }
            GenericArgKind::Const(ct) => matches!(ct.val, ty::ConstKind::Param(_)),
        };

        if arg_is_param {
            seen_params.entry(arg).or_default().push(i);
        } else {
            // Prevent `fn foo() -> Foo<u32>` from being defining.
            let opaque_param = opaque_generics.param_at(i, tcx);
            tcx.sess
                .struct_span_err(span, "non-defining opaque type use in defining scope")
                .span_note(
                    tcx.def_span(opaque_param.def_id),
                    &format!(
                        "used non-generic {} `{}` for generic parameter",
                        opaque_param.kind.descr(),
                        arg,
                    ),
                )
                .emit();
            return false;
        }
    }

    for (_, indices) in seen_params {
        if indices.len() > 1 {
            let descr = opaque_generics.param_at(indices[0], tcx).kind.descr();
            let spans: Vec<_> = indices
                .into_iter()
                .map(|i| tcx.def_span(opaque_generics.param_at(i, tcx).def_id))
                .collect();
            tcx.sess
                .struct_span_err(span, "non-defining opaque type use in defining scope")
                .span_note(spans, &format!("{} used multiple times", descr))
                .emit();
            return false;
        }
    }
    true
}
