//! Orphan checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{Diag, EmissionGuarantee, ErrorGuaranteed};
use rustc_infer::infer::{DefineOpaqueTypes, InferCtxt, TyCtxtInferExt};
use rustc_lint_defs::builtin::UNCOVERED_PARAM_IN_PROJECTION;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeFlags, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
    TypingMode,
};
use rustc_middle::{bug, span_bug};
use rustc_span::def_id::LocalDefId;
use rustc_span::{Ident, Span};
use rustc_trait_selection::traits::{self, InSelfTy, OrphanCheckErr, OrphanCheckMode, UncoveredTy};
use tracing::{debug, instrument};

use crate::errors;

#[instrument(level = "debug", skip(tcx))]
pub(crate) fn orphan_check_impl(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    let trait_ref = tcx.impl_trait_ref(impl_def_id).instantiate_identity();
    trait_ref.error_reported()?;

    match orphan_check(tcx, impl_def_id, OrphanCheckMode::Proper) {
        Ok(()) => {}
        Err(err) => match orphan_check(tcx, impl_def_id, OrphanCheckMode::Compat) {
            Ok(()) => match err {
                OrphanCheckErr::UncoveredTy(UncoveredTy { uncovered, local_ty, in_self_ty }) => {
                    let item = tcx.hir_expect_item(impl_def_id);
                    let impl_ = item.expect_impl();
                    let hir_trait_ref = impl_.of_trait.as_ref().unwrap();

                    // FIXME: Dedupe!
                    // Given `impl<A, B> C<B> for D<A>`,
                    let span = match in_self_ty {
                        InSelfTy::Yes => impl_.self_ty.span,     // point at `D<A>`.
                        InSelfTy::No => hir_trait_ref.path.span, // point at `C<B>`.
                    };

                    for ty in uncovered {
                        match ty {
                            UncoveredTyKind::TyParam(ident) => tcx.node_span_lint(
                                UNCOVERED_PARAM_IN_PROJECTION,
                                item.hir_id(),
                                ident.span,
                                |diag| decorate_uncovered_ty_diag(diag, ident.span, ty, local_ty),
                            ),
                            // FIXME(fmease): This one is hard to explain ^^'
                            UncoveredTyKind::Unknown => {
                                let mut diag = tcx.dcx().struct_span_err(span, "");
                                decorate_uncovered_ty_diag(&mut diag, span, ty, local_ty);
                                diag.emit();
                            }
                            _ => bug!(),
                        }
                    }
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

            ty::Bool
            | ty::Pat(..)
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Str
            | ty::Array(..)
            | ty::Slice(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnPtr(..)
            | ty::Never
            | ty::Tuple(..)
            | ty::UnsafeBinder(_) => (LocalImpl::Allow, NonlocalImpl::DisallowOther),

            ty::FnDef(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Infer(..) => {
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

/// Checks the coherence orphan rules for trait impl given by `impl_def_id`.
#[instrument(level = "debug", skip(tcx), ret)]
fn orphan_check<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_def_id: LocalDefId,
    mode: OrphanCheckMode,
) -> Result<(), OrphanCheckErr<TyCtxt<'tcx>, UncoveredTys<'tcx>>> {
    // We only accept this routine to be invoked on implementations
    // of a trait, not inherent implementations.
    let trait_ref = tcx.impl_trait_ref(impl_def_id);
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
        let errors = ocx.try_evaluate_obligations();
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
        OrphanCheckErr::UncoveredTy(UncoveredTy { uncovered, in_self_ty, local_ty }) => {
            let mut collector =
                UncoveredTyCollector { infcx: &infcx, uncovered: Default::default() };
            uncovered.visit_with(&mut collector);
            if collector.uncovered.is_empty() {
                collector.uncovered.insert(UncoveredTyKind::Unknown);
            }

            OrphanCheckErr::UncoveredTy(UncoveredTy {
                uncovered: collector.uncovered,
                in_self_ty,
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
    err: OrphanCheckErr<TyCtxt<'tcx>, UncoveredTys<'tcx>>,
) -> ErrorGuaranteed {
    let item = tcx.hir_expect_item(impl_def_id);
    let impl_ = item.expect_impl();
    let of_trait = impl_.of_trait.unwrap();

    // Given `impl<A, B> C<B> for D<A>`,
    let impl_trait_ref_span = |in_self_ty| match in_self_ty {
        InSelfTy::Yes => impl_.self_ty.span,          // point at `D<A>`.
        InSelfTy::No => of_trait.trait_ref.path.span, // point at `C<B>`.
    };

    match err {
        OrphanCheckErr::NonLocalInputType(tys) => {
            let span = tcx.def_span(impl_def_id);
            let mut diag = tcx.dcx().create_err(match trait_ref.self_ty().kind() {
                ty::Adt(..) => errors::OnlyCurrentTraits::Outside { span, note: () },
                _ if trait_ref.self_ty().is_primitive() => {
                    errors::OnlyCurrentTraits::Primitive { span, note: () }
                }
                _ => errors::OnlyCurrentTraits::Arbitrary { span, note: () },
            });

            for &(mut ty, in_self_ty) in &tys {
                let span = impl_trait_ref_span(in_self_ty);
                let is_foreign =
                    !of_trait.trait_ref.def_id.is_local() && matches!(in_self_ty, InSelfTy::No);

                ty = tcx.erase_and_anonymize_regions(ty);

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
        OrphanCheckErr::UncoveredTy(UncoveredTy { uncovered, in_self_ty, local_ty }) => {
            let span = impl_trait_ref_span(in_self_ty);

            let mut guar = None;
            for ty in uncovered {
                let span = match ty {
                    UncoveredTyKind::TyParam(ident) => ident.span,
                    _ => span,
                };
                let mut diag = tcx.dcx().struct_span_err(span, "");
                decorate_uncovered_ty_diag(&mut diag, span, ty, local_ty);
                guar.get_or_insert(diag.emit());
            }
            // This should not fail because we know that `uncovered` was non-empty at the point of
            // iteration since it always contains a single `Unknown` if all else fails.
            guar.unwrap()
        }
    }
}

fn decorate_uncovered_ty_diag(
    diag: &mut Diag<'_, impl EmissionGuarantee>,
    span: Span,
    kind: UncoveredTyKind<'_>,
    local_ty: Option<Ty<'_>>,
) {
    let descr = match kind {
        UncoveredTyKind::TyParam(ident) => Some(("type parameter", ident.to_string())),
        UncoveredTyKind::OpaqueTy(ty) => Some(("opaque type", ty.to_string())),
        UncoveredTyKind::Unknown => None,
    };

    diag.code(rustc_errors::E0210);
    diag.span_label(
        span,
        match descr {
            Some((kind, _)) => format!("uncovered {kind}"),
            None => "contains an uncovered type".into(),
        },
    );

    let subject = match &descr {
        Some((kind, ty)) => format!("{kind} `{ty}`"),
        None => "type parameters and opaque types".into(),
    };

    let note = "\
        implementing a foreign trait is only possible if \
        at least one of the types for which it is implemented is local";

    if let Some(local_ty) = local_ty {
        diag.primary_message(format!("{subject} must be covered by another type when it appears before the first local type (`{local_ty}`)"));
        diag.note(format!("{note},\nand no uncovered type parameters or opaque types appear before that first local type"));
        diag.note(
            "in this case, 'before' refers to the following order: \
            `impl<...> ForeignTrait<T1, ..., Tn> for T0`, where `T0` is the first and `Tn` is the last",
        );
    } else {
        let example = descr.map(|(_, ty)| format!(" (e.g., `MyStruct<{ty}>`)")).unwrap_or_default();
        diag.primary_message(format!(
            "{subject} must be used as the argument to some local type{example}"
        ));
        diag.note(note);
        diag.note(
            "only traits defined in the current crate can be implemented for type parameters and opaque types"
        );
    }
}

struct UncoveredTyCollector<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'tcx>,
    uncovered: UncoveredTys<'tcx>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for UncoveredTyCollector<'_, 'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        if !ty.has_type_flags(TypeFlags::HAS_TY_INFER | TypeFlags::HAS_TY_OPAQUE) {
            return;
        }
        match *ty.kind() {
            ty::Infer(ty::TyVar(vid)) => {
                if let Some(def_id) = self.infcx.type_var_origin(vid).param_def_id {
                    let ident = self.infcx.tcx.item_ident(def_id);
                    self.uncovered.insert(UncoveredTyKind::TyParam(ident));
                }
            }
            // This only works with the old solver. With the next solver, alias types like opaque
            // types structurally normalize to an infer var that is "unresolvable" under coherence.
            // Furthermore, the orphan checker returns the unnormalized type in such cases (with
            // exception like for `Fundamental<?opaque>`) which would be Weak for TAITs and
            // Projection for ATPITs.
            // FIXME(fmease): One solution I could see working would be to reintroduce
            //                "TypeVarOriginKind::OpaqueTy(_)" and to stop OrphanChecker from
            //                remapping to the unnormalized type at all.
            // FIXME(fmease): Should we just let uncovered Opaques take precedence over
            //                uncovered TyParams *inside* Opaques?
            ty::Alias(ty::Opaque, alias) if !alias.has_type_flags(TypeFlags::HAS_TY_INFER) => {
                self.uncovered.insert(UncoveredTyKind::OpaqueTy(ty));
            }
            _ => ty.super_visit_with(self),
        }
    }

    fn visit_const(&mut self, ct: ty::Const<'tcx>) -> Self::Result {
        if ct.has_type_flags(TypeFlags::HAS_TY_INFER | TypeFlags::HAS_TY_OPAQUE) {
            ct.super_visit_with(self)
        }
    }
}

type UncoveredTys<'tcx> = FxIndexSet<UncoveredTyKind<'tcx>>;

#[derive(PartialEq, Eq, Hash, Debug)]
enum UncoveredTyKind<'tcx> {
    TyParam(Ident),
    OpaqueTy(Ty<'tcx>),
    Unknown,
}
