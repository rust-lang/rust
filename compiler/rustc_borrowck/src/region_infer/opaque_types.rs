use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::vec_map::VecMap;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::OpaqueTyOrigin;
use rustc_infer::infer::error_reporting::unexpected_hidden_region_diagnostic;
use rustc_infer::infer::TyCtxtInferExt as _;
use rustc_infer::infer::{DefiningAnchor, InferCtxt};
use rustc_infer::traits::{Obligation, ObligationCause, TraitEngine};
use rustc_middle::ty::fold::{TypeFolder, TypeSuperFoldable};
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, InternalSubsts};
use rustc_middle::ty::visit::TypeVisitable;
use rustc_middle::ty::{
    self, OpaqueHiddenType, OpaqueTypeKey, ToPredicate, Ty, TyCtxt, TypeFoldable,
};
use rustc_span::Span;
use rustc_trait_selection::traits::error_reporting::InferCtxtExt as _;
use rustc_trait_selection::traits::TraitEngineExt as _;

use crate::session_diagnostics::ConstNotUsedTraitAlias;

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
    ) -> VecMap<LocalDefId, OpaqueHiddenType<'tcx>> {
        let mut result: VecMap<LocalDefId, OpaqueHiddenType<'tcx>> = VecMap::new();
        for (opaque_type_key, (concrete_type, origin)) in opaque_ty_decls {
            let substs = opaque_type_key.substs;
            debug!(?concrete_type, ?substs);

            let mut subst_regions = vec![self.universal_regions.fr_static];
            let universal_substs = infcx.tcx.fold_regions(substs, |region, _| {
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
                infcx.tcx.fold_regions(concrete_type, |region, _| match *region {
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
            let ty = infcx.infer_opaque_definition_from_instantiation(
                opaque_type_key,
                universal_concrete_type,
                origin,
            );
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
        tcx.fold_regions(ty, |region, _| match *region {
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

pub trait InferCtxtExt<'tcx> {
    fn infer_opaque_definition_from_instantiation(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        instantiated_ty: OpaqueHiddenType<'tcx>,
        origin: OpaqueTyOrigin,
    ) -> Ty<'tcx>;
}

impl<'a, 'tcx> InferCtxtExt<'tcx> for InferCtxt<'a, 'tcx> {
    /// Given the fully resolved, instantiated type for an opaque
    /// type, i.e., the value of an inference variable like C1 or C2
    /// (*), computes the "definition type" for an opaque type
    /// definition -- that is, the inferred value of `Foo1<'x>` or
    /// `Foo2<'x>` that we would conceptually use in its definition:
    /// ```ignore (illustrative)
    /// type Foo1<'x> = impl Bar<'x> = AAA;  // <-- this type AAA
    /// type Foo2<'x> = impl Bar<'x> = BBB;  // <-- or this type BBB
    /// fn foo<'a, 'b>(..) -> (Foo1<'a>, Foo2<'b>) { .. }
    /// ```
    /// Note that these values are defined in terms of a distinct set of
    /// generic parameters (`'x` instead of `'a`) from C1 or C2. The main
    /// purpose of this function is to do that translation.
    ///
    /// (*) C1 and C2 were introduced in the comments on
    /// `register_member_constraints`. Read that comment for more context.
    ///
    /// # Parameters
    ///
    /// - `def_id`, the `impl Trait` type
    /// - `substs`, the substs  used to instantiate this opaque type
    /// - `instantiated_ty`, the inferred type C1 -- fully resolved, lifted version of
    ///   `opaque_defn.concrete_ty`
    #[instrument(level = "debug", skip(self))]
    fn infer_opaque_definition_from_instantiation(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        instantiated_ty: OpaqueHiddenType<'tcx>,
        origin: OpaqueTyOrigin,
    ) -> Ty<'tcx> {
        if self.is_tainted_by_errors() {
            return self.tcx.ty_error();
        }

        let OpaqueTypeKey { def_id, substs } = opaque_type_key;

        // Use substs to build up a reverse map from regions to their
        // identity mappings. This is necessary because of `impl
        // Trait` lifetimes are computed by replacing existing
        // lifetimes with 'static and remapping only those used in the
        // `impl Trait` return type, resulting in the parameters
        // shifting.
        let id_substs = InternalSubsts::identity_for_item(self.tcx, def_id.to_def_id());
        debug!(?id_substs);
        let map: FxHashMap<GenericArg<'tcx>, GenericArg<'tcx>> =
            substs.iter().enumerate().map(|(index, subst)| (subst, id_substs[index])).collect();
        debug!("map = {:#?}", map);

        // Convert the type from the function into a type valid outside
        // the function, by replacing invalid regions with 'static,
        // after producing an error for each of them.
        let definition_ty = instantiated_ty.ty.fold_with(&mut ReverseMapper::new(
            self.tcx,
            opaque_type_key,
            map,
            instantiated_ty.ty,
            instantiated_ty.span,
        ));
        debug!(?definition_ty);

        if !check_opaque_type_parameter_valid(
            self.tcx,
            opaque_type_key,
            origin,
            instantiated_ty.span,
        ) {
            return self.tcx.ty_error();
        }

        // Only check this for TAIT. RPIT already supports `src/test/ui/impl-trait/nested-return-type2.rs`
        // on stable and we'd break that.
        if let OpaqueTyOrigin::TyAlias = origin {
            // This logic duplicates most of `check_opaque_meets_bounds`.
            // FIXME(oli-obk): Also do region checks here and then consider removing `check_opaque_meets_bounds` entirely.
            let param_env = self.tcx.param_env(def_id);
            let body_id = self.tcx.local_def_id_to_hir_id(def_id);
            // HACK This bubble is required for this tests to pass:
            // type-alias-impl-trait/issue-67844-nested-opaque.rs
            self.tcx.infer_ctxt().with_opaque_type_inference(DefiningAnchor::Bubble).enter(
                move |infcx| {
                    // Require the hidden type to be well-formed with only the generics of the opaque type.
                    // Defining use functions may have more bounds than the opaque type, which is ok, as long as the
                    // hidden type is well formed even without those bounds.
                    let predicate =
                        ty::Binder::dummy(ty::PredicateKind::WellFormed(definition_ty.into()))
                            .to_predicate(infcx.tcx);
                    let mut fulfillment_cx = <dyn TraitEngine<'tcx>>::new(infcx.tcx);

                    // Require that the hidden type actually fulfills all the bounds of the opaque type, even without
                    // the bounds that the function supplies.
                    match infcx.register_hidden_type(
                        OpaqueTypeKey { def_id, substs: id_substs },
                        ObligationCause::misc(instantiated_ty.span, body_id),
                        param_env,
                        definition_ty,
                        origin,
                    ) {
                        Ok(infer_ok) => {
                            for obligation in infer_ok.obligations {
                                fulfillment_cx.register_predicate_obligation(&infcx, obligation);
                            }
                        }
                        Err(err) => {
                            infcx
                                .report_mismatched_types(
                                    &ObligationCause::misc(instantiated_ty.span, body_id),
                                    self.tcx.mk_opaque(def_id.to_def_id(), id_substs),
                                    definition_ty,
                                    err,
                                )
                                .emit();
                        }
                    }

                    fulfillment_cx.register_predicate_obligation(
                        &infcx,
                        Obligation::misc(instantiated_ty.span, body_id, param_env, predicate),
                    );

                    // Check that all obligations are satisfied by the implementation's
                    // version.
                    let errors = fulfillment_cx.select_all_or_error(&infcx);

                    // This is still required for many(half of the tests in ui/type-alias-impl-trait)
                    // tests to pass
                    let _ = infcx.inner.borrow_mut().opaque_type_storage.take_opaque_types();

                    if errors.is_empty() {
                        definition_ty
                    } else {
                        infcx.report_fulfillment_errors(&errors, None, false);
                        self.tcx.ty_error()
                    }
                },
            )
        } else {
            definition_ty
        }
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
            GenericArgKind::Const(ct) => matches!(ct.kind(), ty::ConstKind::Param(_)),
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

struct ReverseMapper<'tcx> {
    tcx: TyCtxt<'tcx>,

    key: ty::OpaqueTypeKey<'tcx>,
    map: FxHashMap<GenericArg<'tcx>, GenericArg<'tcx>>,
    map_missing_regions_to_empty: bool,

    /// initially `Some`, set to `None` once error has been reported
    hidden_ty: Option<Ty<'tcx>>,

    /// Span of function being checked.
    span: Span,
}

impl<'tcx> ReverseMapper<'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        key: ty::OpaqueTypeKey<'tcx>,
        map: FxHashMap<GenericArg<'tcx>, GenericArg<'tcx>>,
        hidden_ty: Ty<'tcx>,
        span: Span,
    ) -> Self {
        Self {
            tcx,
            key,
            map,
            map_missing_regions_to_empty: false,
            hidden_ty: Some(hidden_ty),
            span,
        }
    }

    fn fold_kind_mapping_missing_regions_to_empty(
        &mut self,
        kind: GenericArg<'tcx>,
    ) -> GenericArg<'tcx> {
        assert!(!self.map_missing_regions_to_empty);
        self.map_missing_regions_to_empty = true;
        let kind = kind.fold_with(self);
        self.map_missing_regions_to_empty = false;
        kind
    }

    fn fold_kind_normally(&mut self, kind: GenericArg<'tcx>) -> GenericArg<'tcx> {
        assert!(!self.map_missing_regions_to_empty);
        kind.fold_with(self)
    }
}

impl<'tcx> TypeFolder<'tcx> for ReverseMapper<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    #[instrument(skip(self), level = "debug")]
    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            // Ignore bound regions and `'static` regions that appear in the
            // type, we only need to remap regions that reference lifetimes
            // from the function declaration.
            // This would ignore `'r` in a type like `for<'r> fn(&'r u32)`.
            ty::ReLateBound(..) | ty::ReStatic => return r,

            // If regions have been erased (by writeback), don't try to unerase
            // them.
            ty::ReErased => return r,

            // The regions that we expect from borrow checking.
            ty::ReEarlyBound(_) | ty::ReFree(_) | ty::ReEmpty(ty::UniverseIndex::ROOT) => {}

            ty::ReEmpty(_) | ty::RePlaceholder(_) | ty::ReVar(_) => {
                // All of the regions in the type should either have been
                // erased by writeback, or mapped back to named regions by
                // borrow checking.
                bug!("unexpected region kind in opaque type: {:?}", r);
            }
        }

        let generics = self.tcx().generics_of(self.key.def_id);
        match self.map.get(&r.into()).map(|k| k.unpack()) {
            Some(GenericArgKind::Lifetime(r1)) => r1,
            Some(u) => panic!("region mapped to unexpected kind: {:?}", u),
            None if self.map_missing_regions_to_empty => self.tcx.lifetimes.re_root_empty,
            None if generics.parent.is_some() => {
                if let Some(hidden_ty) = self.hidden_ty.take() {
                    unexpected_hidden_region_diagnostic(
                        self.tcx,
                        self.tcx.def_span(self.key.def_id),
                        hidden_ty,
                        r,
                        self.key,
                    )
                    .emit();
                }
                self.tcx.lifetimes.re_root_empty
            }
            None => {
                self.tcx
                    .sess
                    .struct_span_err(self.span, "non-defining opaque type use in defining scope")
                    .span_label(
                        self.span,
                        format!(
                            "lifetime `{}` is part of concrete type but not used in \
                                 parameter list of the `impl Trait` type alias",
                            r
                        ),
                    )
                    .emit();

                self.tcx().lifetimes.re_static
            }
        }
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match *ty.kind() {
            ty::Closure(def_id, substs) => {
                // I am a horrible monster and I pray for death. When
                // we encounter a closure here, it is always a closure
                // from within the function that we are currently
                // type-checking -- one that is now being encapsulated
                // in an opaque type. Ideally, we would
                // go through the types/lifetimes that it references
                // and treat them just like we would any other type,
                // which means we would error out if we find any
                // reference to a type/region that is not in the
                // "reverse map".
                //
                // **However,** in the case of closures, there is a
                // somewhat subtle (read: hacky) consideration. The
                // problem is that our closure types currently include
                // all the lifetime parameters declared on the
                // enclosing function, even if they are unused by the
                // closure itself. We can't readily filter them out,
                // so here we replace those values with `'empty`. This
                // can't really make a difference to the rest of the
                // compiler; those regions are ignored for the
                // outlives relation, and hence don't affect trait
                // selection or auto traits, and they are erased
                // during codegen.

                let generics = self.tcx.generics_of(def_id);
                let substs = self.tcx.mk_substs(substs.iter().enumerate().map(|(index, kind)| {
                    if index < generics.parent_count {
                        // Accommodate missing regions in the parent kinds...
                        self.fold_kind_mapping_missing_regions_to_empty(kind)
                    } else {
                        // ...but not elsewhere.
                        self.fold_kind_normally(kind)
                    }
                }));

                self.tcx.mk_closure(def_id, substs)
            }

            ty::Generator(def_id, substs, movability) => {
                let generics = self.tcx.generics_of(def_id);
                let substs = self.tcx.mk_substs(substs.iter().enumerate().map(|(index, kind)| {
                    if index < generics.parent_count {
                        // Accommodate missing regions in the parent kinds...
                        self.fold_kind_mapping_missing_regions_to_empty(kind)
                    } else {
                        // ...but not elsewhere.
                        self.fold_kind_normally(kind)
                    }
                }));

                self.tcx.mk_generator(def_id, substs, movability)
            }

            ty::Param(param) => {
                // Look it up in the substitution list.
                match self.map.get(&ty.into()).map(|k| k.unpack()) {
                    // Found it in the substitution list; replace with the parameter from the
                    // opaque type.
                    Some(GenericArgKind::Type(t1)) => t1,
                    Some(u) => panic!("type mapped to unexpected kind: {:?}", u),
                    None => {
                        debug!(?param, ?self.map);
                        self.tcx
                            .sess
                            .struct_span_err(
                                self.span,
                                &format!(
                                    "type parameter `{}` is part of concrete type but not \
                                          used in parameter list for the `impl Trait` type alias",
                                    ty
                                ),
                            )
                            .emit();

                        self.tcx().ty_error()
                    }
                }
            }

            _ => ty.super_fold_with(self),
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        trace!("checking const {:?}", ct);
        // Find a const parameter
        match ct.kind() {
            ty::ConstKind::Param(..) => {
                // Look it up in the substitution list.
                match self.map.get(&ct.into()).map(|k| k.unpack()) {
                    // Found it in the substitution list, replace with the parameter from the
                    // opaque type.
                    Some(GenericArgKind::Const(c1)) => c1,
                    Some(u) => panic!("const mapped to unexpected kind: {:?}", u),
                    None => {
                        self.tcx.sess.emit_err(ConstNotUsedTraitAlias {
                            ct: ct.to_string(),
                            span: self.span,
                        });

                        self.tcx().const_error(ct.ty())
                    }
                }
            }

            _ => ct,
        }
    }
}
