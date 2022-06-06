use crate::traits;
use crate::traits::error_reporting::InferCtxtExt as _;
use crate::traits::TraitEngineExt as _;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_hir::OpaqueTyOrigin;
use rustc_infer::infer::error_reporting::unexpected_hidden_region_diagnostic;
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt as _};
use rustc_infer::traits::{Obligation, ObligationCause, TraitEngine};
use rustc_middle::ty::fold::{TypeFoldable, TypeFolder};
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, InternalSubsts};
use rustc_middle::ty::{self, OpaqueHiddenType, OpaqueTypeKey, ToPredicate, Ty, TyCtxt};
use rustc_span::Span;

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
        let id_substs = InternalSubsts::identity_for_item(self.tcx, def_id);
        debug!(?id_substs);
        let map: FxHashMap<GenericArg<'tcx>, GenericArg<'tcx>> =
            substs.iter().enumerate().map(|(index, subst)| (subst, id_substs[index])).collect();
        debug!("map = {:#?}", map);

        // Convert the type from the function into a type valid outside
        // the function, by replacing invalid regions with 'static,
        // after producing an error for each of them.
        let definition_ty = instantiated_ty.ty.fold_with(&mut ReverseMapper::new(
            self.tcx,
            def_id,
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
            let body_id = self.tcx.local_def_id_to_hir_id(def_id.as_local().unwrap());
            self.tcx.infer_ctxt().enter(move |infcx| {
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
                                self.tcx.mk_opaque(def_id, id_substs),
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

                let _ = infcx.inner.borrow_mut().opaque_type_storage.take_opaque_types();

                if errors.is_empty() {
                    definition_ty
                } else {
                    infcx.report_fulfillment_errors(&errors, None, false);
                    self.tcx.ty_error()
                }
            })
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

struct ReverseMapper<'tcx> {
    tcx: TyCtxt<'tcx>,

    opaque_type_def_id: DefId,
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
        opaque_type_def_id: DefId,
        map: FxHashMap<GenericArg<'tcx>, GenericArg<'tcx>>,
        hidden_ty: Ty<'tcx>,
        span: Span,
    ) -> Self {
        Self {
            tcx,
            opaque_type_def_id,
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

        let generics = self.tcx().generics_of(self.opaque_type_def_id);
        match self.map.get(&r.into()).map(|k| k.unpack()) {
            Some(GenericArgKind::Lifetime(r1)) => r1,
            Some(u) => panic!("region mapped to unexpected kind: {:?}", u),
            None if self.map_missing_regions_to_empty => self.tcx.lifetimes.re_root_empty,
            None if generics.parent.is_some() => {
                if let Some(hidden_ty) = self.hidden_ty.take() {
                    unexpected_hidden_region_diagnostic(
                        self.tcx,
                        self.tcx.def_span(self.opaque_type_def_id),
                        hidden_ty,
                        r,
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
        match ct.val() {
            ty::ConstKind::Param(..) => {
                // Look it up in the substitution list.
                match self.map.get(&ct.into()).map(|k| k.unpack()) {
                    // Found it in the substitution list, replace with the parameter from the
                    // opaque type.
                    Some(GenericArgKind::Const(c1)) => c1,
                    Some(u) => panic!("const mapped to unexpected kind: {:?}", u),
                    None => {
                        self.tcx
                            .sess
                            .struct_span_err(
                                self.span,
                                &format!(
                                    "const parameter `{}` is part of concrete type but not \
                                          used in parameter list for the `impl Trait` type alias",
                                    ct
                                ),
                            )
                            .emit();

                        self.tcx().const_error(ct.ty())
                    }
                }
            }

            _ => ct,
        }
    }
}

/// Given a set of predicates that apply to an object type, returns
/// the region bounds that the (erased) `Self` type must
/// outlive. Precisely *because* the `Self` type is erased, the
/// parameter `erased_self_ty` must be supplied to indicate what type
/// has been used to represent `Self` in the predicates
/// themselves. This should really be a unique type; `FreshTy(0)` is a
/// popular choice.
///
/// N.B., in some cases, particularly around higher-ranked bounds,
/// this function returns a kind of conservative approximation.
/// That is, all regions returned by this function are definitely
/// required, but there may be other region bounds that are not
/// returned, as well as requirements like `for<'a> T: 'a`.
///
/// Requires that trait definitions have been processed so that we can
/// elaborate predicates and walk supertraits.
#[instrument(skip(tcx, predicates), level = "debug")]
pub(crate) fn required_region_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    erased_self_ty: Ty<'tcx>,
    predicates: impl Iterator<Item = ty::Predicate<'tcx>>,
) -> Vec<ty::Region<'tcx>> {
    assert!(!erased_self_ty.has_escaping_bound_vars());

    traits::elaborate_predicates(tcx, predicates)
        .filter_map(|obligation| {
            debug!(?obligation);
            match obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Projection(..)
                | ty::PredicateKind::Trait(..)
                | ty::PredicateKind::Subtype(..)
                | ty::PredicateKind::Coerce(..)
                | ty::PredicateKind::WellFormed(..)
                | ty::PredicateKind::ObjectSafe(..)
                | ty::PredicateKind::ClosureKind(..)
                | ty::PredicateKind::RegionOutlives(..)
                | ty::PredicateKind::ConstEvaluatable(..)
                | ty::PredicateKind::ConstEquate(..)
                | ty::PredicateKind::TypeWellFormedFromEnv(..) => None,
                ty::PredicateKind::TypeOutlives(ty::OutlivesPredicate(ref t, ref r)) => {
                    // Search for a bound of the form `erased_self_ty
                    // : 'a`, but be wary of something like `for<'a>
                    // erased_self_ty : 'a` (we interpret a
                    // higher-ranked bound like that as 'static,
                    // though at present the code in `fulfill.rs`
                    // considers such bounds to be unsatisfiable, so
                    // it's kind of a moot point since you could never
                    // construct such an object, but this seems
                    // correct even if that code changes).
                    if t == &erased_self_ty && !r.has_escaping_bound_vars() {
                        Some(*r)
                    } else {
                        None
                    }
                }
            }
        })
        .collect()
}
