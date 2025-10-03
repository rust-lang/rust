//! Orphan checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::ErrorGuaranteed;
use rustc_infer::infer::{DefineOpaqueTypes, InferCtxt, TyCtxtInferExt};
use rustc_lint_defs::builtin::UNCOVERED_PARAM_IN_PROJECTION;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor, TypingMode,
};
use rustc_middle::{bug, span_bug};
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_trait_selection::traits::{
    self, IsFirstInputType, OrphanCheckErr, OrphanCheckMode, UncoveredTyParams,
};
use tracing::{debug, instrument};

use crate::errors;

#[instrument(level = "debug", skip(tcx))]
pub(crate) fn orphan_check_impl(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap().instantiate_identity();
    trait_ref.error_reported()?;

    match orphan_check(tcx, impl_def_id, OrphanCheckMode::Proper) {
        Ok(()) => {}
        Err(err) => match orphan_check(tcx, impl_def_id, OrphanCheckMode::Compat) {
            Ok(()) => match err {
                OrphanCheckErr::UncoveredTyParams(uncovered_ty_params) => {
                    lint_uncovered_ty_params(tcx, uncovered_ty_params, impl_def_id)
                }
                OrphanCheckErr::NonLocalInputType(_) => {
                    bug!("orphanck: shouldn't've gotten non-local input tys in compat mode")
                }
            },
            Err(err) => return Err(emit_orphan_check_error(tcx, trait_ref, impl_def_id, err)),
        },
    }

    let trait_def_id = trait_ref.def_id;

    // In addition to the above rules, we restrict impls of auto traits
    // so that they can only be implemented on nominal types, such as structs,
    // enums or foreign types. To see why this restriction exists, consider the
    // following example (#22978). Imagine that crate A defines an auto trait
    // `Foo` and a fn that operates on pairs of types:
    //
    // ```
    // // Crate A
    // auto trait Foo { }
    // fn two_foos<A:Foo,B:Foo>(..) {
    //     one_foo::<(A,B)>(..)
    // }
    // fn one_foo<T:Foo>(..) { .. }
    // ```
    //
    // This type-checks fine; in particular the fn
    // `two_foos` is able to conclude that `(A,B):Foo`
    // because `A:Foo` and `B:Foo`.
    //
    // Now imagine that crate B comes along and does the following:
    //
    // ```
    // struct A { }
    // struct B { }
    // impl Foo for A { }
    // impl Foo for B { }
    // impl !Foo for (A, B) { }
    // ```
    //
    // This final impl is legal according to the orphan
    // rules, but it invalidates the reasoning from
    // `two_foos` above.
    debug!(
        "trait_ref={:?} trait_def_id={:?} trait_is_auto={}",
        trait_ref,
        trait_def_id,
        tcx.trait_is_auto(trait_def_id)
    );

    if tcx.trait_is_auto(trait_def_id) {
        let self_ty = trait_ref.self_ty();

        // If the impl is in the same crate as the auto-trait, almost anything
        // goes.
        //
        //     impl MyAuto for Rc<Something> {}  // okay
        //     impl<T> !MyAuto for *const T {}   // okay
        //     impl<T> MyAuto for T {}           // okay
        //
        // But there is one important exception: implementing for a trait object
        // is not allowed.
        //
        //     impl MyAuto for dyn Trait {}      // NOT OKAY
        //     impl<T: ?Sized> MyAuto for T {}   // NOT OKAY
        //
        // With this restriction, it's guaranteed that an auto-trait is
        // implemented for a trait object if and only if the auto-trait is one
        // of the trait object's trait bounds (or a supertrait of a bound). In
        // other words `dyn Trait + AutoTrait` always implements AutoTrait,
        // while `dyn Trait` never implements AutoTrait.
        //
        // This is necessary in order for autotrait bounds on methods of trait
        // objects to be sound.
        //
        //     auto trait AutoTrait {}
        //
        //     trait DynCompatibleTrait {
        //         fn f(&self) where Self: AutoTrait;
        //     }
        //
        // We can allow f to be called on `dyn DynCompatibleTrait + AutoTrait`.
        //
        // If we didn't deny `impl AutoTrait for dyn Trait`, it would be unsound
        // for the `DynCompatibleTrait` shown above to be dyn-compatible because someone
        // could take some type implementing `DynCompatibleTrait` but not `AutoTrait`,
        // unsize it to `dyn DynCompatibleTrait`, and call `.f()` which has no
        // concrete implementation (issue #50781).
        enum LocalImpl {
            Allow,
            Disallow { problematic_kind: &'static str },
        }

        // If the auto-trait is from a dependency, it must only be getting
        // implemented for a nominal type, and specifically one local to the
        // current crate.
        //
        //     impl<T> Sync for MyStruct<T> {}   // okay
        //
        //     impl Sync for Rc<MyStruct> {}     // NOT OKAY
        enum NonlocalImpl {
            Allow,
            DisallowBecauseNonlocal,
            DisallowOther,
        }

        // Exhaustive match considering that this logic is essential for
        // soundness.
        let (local_impl, nonlocal_impl) = match self_ty.kind() {
            // struct Struct<T>;
            // impl AutoTrait for Struct<Foo> {}
            ty::Adt(self_def, _) => (
                LocalImpl::Allow,
                if self_def.did().is_local() {
                    NonlocalImpl::Allow
                } else {
                    NonlocalImpl::DisallowBecauseNonlocal
                },
            ),

            // extern { type OpaqueType; }
            // impl AutoTrait for OpaqueType {}
            ty::Foreign(did) => (
                LocalImpl::Allow,
                if did.is_local() {
                    NonlocalImpl::Allow
                } else {
                    NonlocalImpl::DisallowBecauseNonlocal
                },
            ),

            // impl AutoTrait for dyn Trait {}
            ty::Dynamic(..) => (
                LocalImpl::Disallow { problematic_kind: "trait object" },
                NonlocalImpl::DisallowOther,
            ),

            // impl<T> AutoTrait for T {}
            // impl<T: ?Sized> AutoTrait for T {}
            ty::Param(..) => (
                if self_ty.is_sized(tcx, ty::TypingEnv::non_body_analysis(tcx, impl_def_id)) {
                    LocalImpl::Allow
                } else {
                    LocalImpl::Disallow { problematic_kind: "generic type" }
                },
                NonlocalImpl::DisallowOther,
            ),

            ty::Alias(kind, _) => {
                let problematic_kind = match kind {
                    // trait Id { type This: ?Sized; }
                    // impl<T: ?Sized> Id for T {
                    //     type This = T;
                    // }
                    // impl<T: ?Sized> AutoTrait for <T as Id>::This {}
                    ty::Projection => "associated type",
                    // type Foo = (impl Sized, bool)
                    // impl AutoTrait for Foo {}
                    ty::Free => "type alias",
                    // type Opaque = impl Trait;
                    // impl AutoTrait for Opaque {}
                    ty::Opaque => "opaque type",
                    // ```
                    // struct S<T>(T);
                    // impl<T: ?Sized> S<T> {
                    //     type This = T;
                    // }
                    // impl<T: ?Sized> AutoTrait for S<T>::This {}
                    // ```
                    // FIXME(inherent_associated_types): The example code above currently leads to a cycle
                    ty::Inherent => "associated type",
                };
                (LocalImpl::Disallow { problematic_kind }, NonlocalImpl::DisallowOther)
            }

            ty::Pat(..) => (
                LocalImpl::Disallow { problematic_kind: "pattern type" },
                NonlocalImpl::DisallowOther,
            ),

            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Str
            | ty::Array(..)
            | ty::Slice(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Never
            | ty::Tuple(..)
            | ty::UnsafeBinder(_) => (LocalImpl::Allow, NonlocalImpl::DisallowOther),

            ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..) => {
                return Err(tcx
                    .dcx()
                    .delayed_bug("cannot define inherent `impl` for closure types"));
            }
            ty::Bound(..) | ty::Placeholder(..) | ty::Infer(..) => {
                let sp = tcx.def_span(impl_def_id);
                span_bug!(sp, "weird self type for autotrait impl")
            }

            ty::Error(..) => (LocalImpl::Allow, NonlocalImpl::Allow),
        };

        if trait_def_id.is_local() {
            match local_impl {
                LocalImpl::Allow => {}
                LocalImpl::Disallow { problematic_kind } => {
                    return Err(tcx.dcx().emit_err(errors::TraitsWithDefaultImpl {
                        span: tcx.def_span(impl_def_id),
                        traits: tcx.def_path_str(trait_def_id),
                        problematic_kind,
                        self_ty,
                    }));
                }
            }
        } else {
            match nonlocal_impl {
                NonlocalImpl::Allow => {}
                NonlocalImpl::DisallowBecauseNonlocal => {
                    return Err(tcx.dcx().emit_err(errors::CrossCrateTraitsDefined {
                        span: tcx.def_span(impl_def_id),
                        traits: tcx.def_path_str(trait_def_id),
                    }));
                }
                NonlocalImpl::DisallowOther => {
                    return Err(tcx.dcx().emit_err(errors::CrossCrateTraits {
                        span: tcx.def_span(impl_def_id),
                        traits: tcx.def_path_str(trait_def_id),
                        self_ty,
                    }));
                }
            }
        }
    }

    Ok(())
}

/// Checks the coherence orphan rules.
///
/// `impl_def_id` should be the `DefId` of a trait impl.
///
/// To pass, either the trait must be local, or else two conditions must be satisfied:
///
/// 1. All type parameters in `Self` must be "covered" by some local type constructor.
/// 2. Some local type must appear in `Self`.
#[instrument(level = "debug", skip(tcx), ret)]
fn orphan_check<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_def_id: LocalDefId,
    mode: OrphanCheckMode,
) -> Result<(), OrphanCheckErr<TyCtxt<'tcx>, FxIndexSet<DefId>>> {
    // We only accept this routine to be invoked on implementations
    // of a trait, not inherent implementations.
    let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap();
    debug!(trait_ref = ?trait_ref.skip_binder());

    // If the *trait* is local to the crate, ok.
    if let Some(def_id) = trait_ref.skip_binder().def_id.as_local() {
        debug!("trait {def_id:?} is local to current crate");
        return Ok(());
    }

    // (1)  Instantiate all generic params with fresh inference vars.
    let infcx = tcx.infer_ctxt().build(TypingMode::Coherence);
    let cause = traits::ObligationCause::dummy();
    let args = infcx.fresh_args_for_item(cause.span, impl_def_id.to_def_id());
    let trait_ref = trait_ref.instantiate(tcx, args);

    let lazily_normalize_ty = |user_ty: Ty<'tcx>| {
        let ty::Alias(..) = user_ty.kind() else { return Ok(user_ty) };

        let ocx = traits::ObligationCtxt::new(&infcx);
        let ty = ocx.normalize(&cause, ty::ParamEnv::empty(), user_ty);
        let ty = infcx.resolve_vars_if_possible(ty);
        let errors = ocx.select_where_possible();
        if !errors.is_empty() {
            return Ok(user_ty);
        }

        let ty = if infcx.next_trait_solver() {
            ocx.structurally_normalize_ty(
                &cause,
                ty::ParamEnv::empty(),
                infcx.resolve_vars_if_possible(ty),
            )
            .unwrap_or(ty)
        } else {
            ty
        };

        Ok::<_, !>(ty)
    };

    let result = traits::orphan_check_trait_ref(
        &infcx,
        trait_ref,
        traits::InCrate::Local { mode },
        lazily_normalize_ty,
    )
    .into_ok();

    // (2)  Try to map the remaining inference vars back to generic params.
    result.map_err(|err| match err {
        OrphanCheckErr::UncoveredTyParams(UncoveredTyParams { uncovered, local_ty }) => {
            let mut collector =
                UncoveredTyParamCollector { infcx: &infcx, uncovered_params: Default::default() };
            uncovered.visit_with(&mut collector);
            // FIXME(fmease): This is very likely reachable.
            debug_assert!(!collector.uncovered_params.is_empty());

            OrphanCheckErr::UncoveredTyParams(UncoveredTyParams {
                uncovered: collector.uncovered_params,
                local_ty,
            })
        }
        OrphanCheckErr::NonLocalInputType(tys) => {
            let tys = infcx.probe(|_| {
                // Map the unconstrained args back to their params,
                // ignoring any type unification errors.
                for (arg, id_arg) in
                    std::iter::zip(args, ty::GenericArgs::identity_for_item(tcx, impl_def_id))
                {
                    let _ = infcx.at(&cause, ty::ParamEnv::empty()).eq(
                        DefineOpaqueTypes::No,
                        arg,
                        id_arg,
                    );
                }
                infcx.resolve_vars_if_possible(tys)
            });
            OrphanCheckErr::NonLocalInputType(tys)
        }
    })
}

fn emit_orphan_check_error<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    impl_def_id: LocalDefId,
    err: traits::OrphanCheckErr<TyCtxt<'tcx>, FxIndexSet<DefId>>,
) -> ErrorGuaranteed {
    match err {
        traits::OrphanCheckErr::NonLocalInputType(tys) => {
            let item = tcx.hir_expect_item(impl_def_id);
            let impl_ = item.expect_impl();
            let of_trait = impl_.of_trait.unwrap();

            let span = tcx.def_span(impl_def_id);
            let mut diag = tcx.dcx().create_err(match trait_ref.self_ty().kind() {
                ty::Adt(..) => errors::OnlyCurrentTraits::Outside { span, note: () },
                _ if trait_ref.self_ty().is_primitive() => {
                    errors::OnlyCurrentTraits::Primitive { span, note: () }
                }
                _ => errors::OnlyCurrentTraits::Arbitrary { span, note: () },
            });

            for &(mut ty, is_target_ty) in &tys {
                let span = if matches!(is_target_ty, IsFirstInputType::Yes) {
                    // Point at `D<A>` in `impl<A, B> for C<B> in D<A>`
                    impl_.self_ty.span
                } else {
                    // Point at `C<B>` in `impl<A, B> for C<B> in D<A>`
                    of_trait.trait_ref.path.span
                };

                ty = tcx.erase_and_anonymize_regions(ty);

                let is_foreign =
                    !trait_ref.def_id.is_local() && matches!(is_target_ty, IsFirstInputType::No);

                match *ty.kind() {
                    ty::Slice(_) => {
                        if is_foreign {
                            diag.subdiagnostic(errors::OnlyCurrentTraitsForeign { span });
                        } else {
                            diag.subdiagnostic(errors::OnlyCurrentTraitsName {
                                span,
                                name: "slices",
                            });
                        }
                    }
                    ty::Array(..) => {
                        if is_foreign {
                            diag.subdiagnostic(errors::OnlyCurrentTraitsForeign { span });
                        } else {
                            diag.subdiagnostic(errors::OnlyCurrentTraitsName {
                                span,
                                name: "arrays",
                            });
                        }
                    }
                    ty::Tuple(..) => {
                        if is_foreign {
                            diag.subdiagnostic(errors::OnlyCurrentTraitsForeign { span });
                        } else {
                            diag.subdiagnostic(errors::OnlyCurrentTraitsName {
                                span,
                                name: "tuples",
                            });
                        }
                    }
                    ty::Alias(ty::Opaque, ..) => {
                        diag.subdiagnostic(errors::OnlyCurrentTraitsOpaque { span });
                    }
                    ty::RawPtr(ptr_ty, mutbl) => {
                        if !trait_ref.self_ty().has_param() {
                            diag.subdiagnostic(errors::OnlyCurrentTraitsPointerSugg {
                                wrapper_span: impl_.self_ty.span,
                                struct_span: item.span.shrink_to_lo(),
                                mut_key: mutbl.prefix_str(),
                                ptr_ty,
                            });
                        }
                        diag.subdiagnostic(errors::OnlyCurrentTraitsPointer { span, pointer: ty });
                    }
                    ty::Adt(adt_def, _) => {
                        diag.subdiagnostic(errors::OnlyCurrentTraitsAdt {
                            span,
                            name: tcx.def_path_str(adt_def.did()),
                        });
                    }
                    _ => {
                        diag.subdiagnostic(errors::OnlyCurrentTraitsTy { span, ty });
                    }
                }
            }

            diag.emit()
        }
        traits::OrphanCheckErr::UncoveredTyParams(UncoveredTyParams { uncovered, local_ty }) => {
            let mut reported = None;
            for param_def_id in uncovered {
                let name = tcx.item_ident(param_def_id);
                let span = name.span;

                reported.get_or_insert(match local_ty {
                    Some(local_type) => tcx.dcx().emit_err(errors::TyParamFirstLocal {
                        span,
                        note: (),
                        param: name,
                        local_type,
                    }),
                    None => tcx.dcx().emit_err(errors::TyParamSome { span, note: (), param: name }),
                });
            }
            reported.unwrap() // FIXME(fmease): This is very likely reachable.
        }
    }
}

fn lint_uncovered_ty_params<'tcx>(
    tcx: TyCtxt<'tcx>,
    UncoveredTyParams { uncovered, local_ty }: UncoveredTyParams<TyCtxt<'tcx>, FxIndexSet<DefId>>,
    impl_def_id: LocalDefId,
) {
    let hir_id = tcx.local_def_id_to_hir_id(impl_def_id);

    for param_def_id in uncovered {
        let span = tcx.def_ident_span(param_def_id).unwrap();
        let name = tcx.item_ident(param_def_id);

        match local_ty {
            Some(local_type) => tcx.emit_node_span_lint(
                UNCOVERED_PARAM_IN_PROJECTION,
                hir_id,
                span,
                errors::TyParamFirstLocalLint { span, note: (), param: name, local_type },
            ),
            None => tcx.emit_node_span_lint(
                UNCOVERED_PARAM_IN_PROJECTION,
                hir_id,
                span,
                errors::TyParamSomeLint { span, note: (), param: name },
            ),
        };
    }
}

struct UncoveredTyParamCollector<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'tcx>,
    uncovered_params: FxIndexSet<DefId>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for UncoveredTyParamCollector<'_, 'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        if !ty.has_type_flags(ty::TypeFlags::HAS_TY_INFER) {
            return;
        }
        let ty::Infer(ty::TyVar(vid)) = *ty.kind() else {
            return ty.super_visit_with(self);
        };
        let origin = self.infcx.type_var_origin(vid);
        if let Some(def_id) = origin.param_def_id {
            self.uncovered_params.insert(def_id);
        }
    }

    fn visit_const(&mut self, ct: ty::Const<'tcx>) -> Self::Result {
        if ct.has_type_flags(ty::TypeFlags::HAS_TY_INFER) {
            ct.super_visit_with(self)
        }
    }
}
