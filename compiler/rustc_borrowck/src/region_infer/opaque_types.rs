use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::vec_map::VecMap;
use rustc_hir::def_id::DefId;
use rustc_hir::OpaqueTyOrigin;
use rustc_infer::infer::InferCtxt;
use rustc_middle::ty::subst::GenericArgKind;
use rustc_middle::ty::{self, OpaqueHiddenType, OpaqueTypeKey, TyCtxt, TypeFoldable};
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
        opaque_ty_decls: VecMap<OpaqueTypeKey<'tcx>, (OpaqueHiddenType<'tcx>, OpaqueTyOrigin)>,
    ) -> VecMap<DefId, OpaqueHiddenType<'tcx>> {
        let mut result: VecMap<DefId, OpaqueHiddenType<'tcx>> = VecMap::new();
        for (opaque_type_key, (concrete_type, origin)) in opaque_ty_decls {
            let substs = opaque_type_key.substs;
            debug!(?concrete_type, ?substs);

            let mut subst_regions = vec![self.universal_regions.fr_static];
            let universal_substs = infcx.tcx.fold_regions(substs, &mut false, |region, _| {
                if let ty::RePlaceholder(..) = region.kind() {
                    // Higher kinded regions don't need remapping, they don't refer to anything outside of this the substs.
                    return region;
                }
                let vid = self.to_region_vid(region);
                trace!(?vid);
                let scc = self.constraint_sccs.scc(vid);
                trace!(?scc);
                match self.scc_values.universal_regions_outlived_by(scc).find_map(|lb| {
                    self.eval_equal(vid, lb).then_some(self.definitions[lb].external_name?)
                }) {
                    Some(region) => {
                        let vid = self.universal_regions.to_region_vid(region);
                        subst_regions.push(vid);
                        region
                    }
                    None => {
                        subst_regions.push(vid);
                        infcx.tcx.sess.delay_span_bug(
                            concrete_type.span,
                            "opaque type with non-universal region substs",
                        );
                        infcx.tcx.lifetimes.re_static
                    }
                }
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
            );
            let ty = if check_opaque_type_parameter_valid(
                infcx.tcx,
                opaque_type_key,
                origin,
                concrete_type.span,
            ) {
                remapped_type
            } else {
                infcx.tcx.ty_error()
            };
            // Sometimes two opaque types are the same only after we remap the generic parameters
            // back to the opaque type definition. E.g. we may have `OpaqueType<X, Y>` mapped to `(X, Y)`
            // and `OpaqueType<Y, X>` mapped to `(Y, X)`, and those are the same, but we only know that
            // once we convert the generic parameters to those of the opaque type.
            if let Some(prev) = result.get_mut(&opaque_type_key.def_id) {
                if prev.ty != ty {
                    if !ty.references_error() {
                        prev.report_mismatch(
                            &OpaqueHiddenType { ty, span: concrete_type.span },
                            infcx.tcx,
                        );
                    }
                    prev.ty = infcx.tcx.ty_error();
                }
                // Pick a better span if there is one.
                // FIXME(oli-obk): collect multiple spans for better diagnostics down the road.
                prev.span = prev.span.substitute_dummy(concrete_type.span);
            } else {
                result.insert(
                    opaque_type_key.def_id,
                    OpaqueHiddenType { ty, span: concrete_type.span },
                );
            }
        }
        result
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
                                Some(region) if region.is_static() => {}
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
    origin: OpaqueTyOrigin,
    span: Span,
) -> bool {
    match origin {
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
    let opaque_generics = tcx.generics_of(opaque_type_key.def_id);
    let mut seen_params: FxHashMap<_, Vec<_>> = FxHashMap::default();
    for (i, arg) in opaque_type_key.substs.iter().enumerate() {
        let arg_is_param = match arg.unpack() {
            GenericArgKind::Type(ty) => matches!(ty.kind(), ty::Param(_)),
            GenericArgKind::Lifetime(lt) if lt.is_static() => {
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
                matches!(*lt, ty::ReEarlyBound(_) | ty::ReFree(_))
            }
            GenericArgKind::Const(ct) => matches!(ct.val(), ty::ConstKind::Param(_)),
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
