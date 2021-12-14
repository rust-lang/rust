//! Check properties that are required by built-in traits and set
//! up data structures required by type-checking/codegen.

use crate::errors::{CopyImplOnNonAdt, CopyImplOnTypeWithDtor, DropImplOnWrongItem};
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_hir::ItemKind;
use rustc_infer::infer;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{RegionckMode, TyCtxtInferExt};
use rustc_middle::ty::adjustment::CoerceUnsizedInfo;
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_trait_selection::traits::error_reporting::InferCtxtExt;
use rustc_trait_selection::traits::misc::{can_type_implement_copy, CopyImplementationError};
use rustc_trait_selection::traits::predicate_for_trait_def;
use rustc_trait_selection::traits::{self, ObligationCause, TraitEngine, TraitEngineExt};

pub fn check_trait(tcx: TyCtxt<'_>, trait_def_id: DefId) {
    let lang_items = tcx.lang_items();
    Checker { tcx, trait_def_id }
        .check(lang_items.drop_trait(), visit_implementation_of_drop)
        .check(lang_items.copy_trait(), visit_implementation_of_copy)
        .check(lang_items.coerce_unsized_trait(), visit_implementation_of_coerce_unsized)
        .check(lang_items.dispatch_from_dyn_trait(), visit_implementation_of_dispatch_from_dyn);
}

struct Checker<'tcx> {
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
}

impl<'tcx> Checker<'tcx> {
    fn check<F>(&self, trait_def_id: Option<DefId>, mut f: F) -> &Self
    where
        F: FnMut(TyCtxt<'tcx>, LocalDefId),
    {
        if Some(self.trait_def_id) == trait_def_id {
            for &impl_def_id in self.tcx.hir().trait_impls(self.trait_def_id) {
                f(self.tcx, impl_def_id);
            }
        }
        self
    }
}

fn visit_implementation_of_drop(tcx: TyCtxt<'_>, impl_did: LocalDefId) {
    // Destructors only work on nominal types.
    if let ty::Adt(..) | ty::Error(_) = tcx.type_of(impl_did).kind() {
        return;
    }

    let sp = match tcx.hir().expect_item(impl_did).kind {
        ItemKind::Impl(ref impl_) => impl_.self_ty.span,
        _ => bug!("expected Drop impl item"),
    };

    tcx.sess.emit_err(DropImplOnWrongItem { span: sp });
}

fn visit_implementation_of_copy(tcx: TyCtxt<'_>, impl_did: LocalDefId) {
    debug!("visit_implementation_of_copy: impl_did={:?}", impl_did);

    let impl_hir_id = tcx.hir().local_def_id_to_hir_id(impl_did);

    let self_type = tcx.type_of(impl_did);
    debug!("visit_implementation_of_copy: self_type={:?} (bound)", self_type);

    let span = tcx.hir().span(impl_hir_id);
    let param_env = tcx.param_env(impl_did);
    assert!(!self_type.has_escaping_bound_vars());

    debug!("visit_implementation_of_copy: self_type={:?} (free)", self_type);

    match can_type_implement_copy(tcx, param_env, self_type) {
        Ok(()) => {}
        Err(CopyImplementationError::InfrigingFields(fields)) => {
            let item = tcx.hir().expect_item(impl_did);
            let span = if let ItemKind::Impl(hir::Impl { of_trait: Some(ref tr), .. }) = item.kind {
                tr.path.span
            } else {
                span
            };

            let mut err = struct_span_err!(
                tcx.sess,
                span,
                E0204,
                "the trait `Copy` may not be implemented for this type"
            );
            for span in fields.iter().map(|f| tcx.def_span(f.did)) {
                err.span_label(span, "this field does not implement `Copy`");
            }
            err.emit()
        }
        Err(CopyImplementationError::NotAnAdt) => {
            let item = tcx.hir().expect_item(impl_did);
            let span =
                if let ItemKind::Impl(ref impl_) = item.kind { impl_.self_ty.span } else { span };

            tcx.sess.emit_err(CopyImplOnNonAdt { span });
        }
        Err(CopyImplementationError::HasDestructor) => {
            tcx.sess.emit_err(CopyImplOnTypeWithDtor { span });
        }
    }
}

fn visit_implementation_of_coerce_unsized<'tcx>(tcx: TyCtxt<'tcx>, impl_did: LocalDefId) {
    debug!("visit_implementation_of_coerce_unsized: impl_did={:?}", impl_did);

    // Just compute this for the side-effects, in particular reporting
    // errors; other parts of the code may demand it for the info of
    // course.
    let span = tcx.def_span(impl_did);
    tcx.at(span).coerce_unsized_info(impl_did);
}

fn visit_implementation_of_dispatch_from_dyn<'tcx>(tcx: TyCtxt<'tcx>, impl_did: LocalDefId) {
    debug!("visit_implementation_of_dispatch_from_dyn: impl_did={:?}", impl_did);

    let impl_hir_id = tcx.hir().local_def_id_to_hir_id(impl_did);
    let span = tcx.hir().span(impl_hir_id);

    let dispatch_from_dyn_trait = tcx.require_lang_item(LangItem::DispatchFromDyn, Some(span));

    let source = tcx.type_of(impl_did);
    assert!(!source.has_escaping_bound_vars());
    let target = {
        let trait_ref = tcx.impl_trait_ref(impl_did).unwrap();
        assert_eq!(trait_ref.def_id, dispatch_from_dyn_trait);

        trait_ref.substs.type_at(1)
    };

    debug!("visit_implementation_of_dispatch_from_dyn: {:?} -> {:?}", source, target);

    let param_env = tcx.param_env(impl_did);

    let create_err = |msg: &str| struct_span_err!(tcx.sess, span, E0378, "{}", msg);

    tcx.infer_ctxt().enter(|infcx| {
        let cause = ObligationCause::misc(span, impl_hir_id);

        use ty::TyKind::*;
        match (source.kind(), target.kind()) {
            (&Ref(r_a, _, mutbl_a), Ref(r_b, _, mutbl_b))
                if infcx.at(&cause, param_env).eq(r_a, r_b).is_ok() && mutbl_a == *mutbl_b => {}
            (&RawPtr(tm_a), &RawPtr(tm_b)) if tm_a.mutbl == tm_b.mutbl => (),
            (&Adt(def_a, substs_a), &Adt(def_b, substs_b))
                if def_a.is_struct() && def_b.is_struct() =>
            {
                if def_a != def_b {
                    let source_path = tcx.def_path_str(def_a.did);
                    let target_path = tcx.def_path_str(def_b.did);

                    create_err(&format!(
                        "the trait `DispatchFromDyn` may only be implemented \
                                for a coercion between structures with the same \
                                definition; expected `{}`, found `{}`",
                        source_path, target_path,
                    ))
                    .emit();

                    return;
                }

                if def_a.repr.c() || def_a.repr.packed() {
                    create_err(
                        "structs implementing `DispatchFromDyn` may not have \
                             `#[repr(packed)]` or `#[repr(C)]`",
                    )
                    .emit();
                }

                let fields = &def_a.non_enum_variant().fields;

                let coerced_fields = fields
                    .iter()
                    .filter(|field| {
                        let ty_a = field.ty(tcx, substs_a);
                        let ty_b = field.ty(tcx, substs_b);

                        if let Ok(layout) = tcx.layout_of(param_env.and(ty_a)) {
                            if layout.is_zst() && layout.align.abi.bytes() == 1 {
                                // ignore ZST fields with alignment of 1 byte
                                return false;
                            }
                        }

                        if let Ok(ok) = infcx.at(&cause, param_env).eq(ty_a, ty_b) {
                            if ok.obligations.is_empty() {
                                create_err(
                                    "the trait `DispatchFromDyn` may only be implemented \
                                     for structs containing the field being coerced, \
                                     ZST fields with 1 byte alignment, and nothing else",
                                )
                                .note(&format!(
                                    "extra field `{}` of type `{}` is not allowed",
                                    field.ident, ty_a,
                                ))
                                .emit();

                                return false;
                            }
                        }

                        return true;
                    })
                    .collect::<Vec<_>>();

                if coerced_fields.is_empty() {
                    create_err(
                        "the trait `DispatchFromDyn` may only be implemented \
                            for a coercion between structures with a single field \
                            being coerced, none found",
                    )
                    .emit();
                } else if coerced_fields.len() > 1 {
                    create_err(
                        "implementing the `DispatchFromDyn` trait requires multiple coercions",
                    )
                    .note(
                        "the trait `DispatchFromDyn` may only be implemented \
                                for a coercion between structures with a single field \
                                being coerced",
                    )
                    .note(&format!(
                        "currently, {} fields need coercions: {}",
                        coerced_fields.len(),
                        coerced_fields
                            .iter()
                            .map(|field| {
                                format!(
                                    "`{}` (`{}` to `{}`)",
                                    field.ident,
                                    field.ty(tcx, substs_a),
                                    field.ty(tcx, substs_b),
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                    .emit();
                } else {
                    let mut fulfill_cx = <dyn TraitEngine<'_>>::new(infcx.tcx);

                    for field in coerced_fields {
                        let predicate = predicate_for_trait_def(
                            tcx,
                            param_env,
                            cause.clone(),
                            dispatch_from_dyn_trait,
                            0,
                            field.ty(tcx, substs_a),
                            &[field.ty(tcx, substs_b).into()],
                        );

                        fulfill_cx.register_predicate_obligation(&infcx, predicate);
                    }

                    // Check that all transitive obligations are satisfied.
                    let errors = fulfill_cx.select_all_or_error(&infcx);
                    if !errors.is_empty() {
                        infcx.report_fulfillment_errors(&errors, None, false);
                    }

                    // Finally, resolve all regions.
                    let outlives_env = OutlivesEnvironment::new(param_env);
                    infcx.resolve_regions_and_report_errors(
                        impl_did.to_def_id(),
                        &outlives_env,
                        RegionckMode::default(),
                    );
                }
            }
            _ => {
                create_err(
                    "the trait `DispatchFromDyn` may only be implemented \
                        for a coercion between structures",
                )
                .emit();
            }
        }
    })
}

pub fn coerce_unsized_info<'tcx>(tcx: TyCtxt<'tcx>, impl_did: DefId) -> CoerceUnsizedInfo {
    debug!("compute_coerce_unsized_info(impl_did={:?})", impl_did);

    // this provider should only get invoked for local def-ids
    let impl_did = impl_did.expect_local();
    let span = tcx.def_span(impl_did);

    let coerce_unsized_trait = tcx.require_lang_item(LangItem::CoerceUnsized, Some(span));

    let unsize_trait = tcx.lang_items().require(LangItem::Unsize).unwrap_or_else(|err| {
        tcx.sess.fatal(&format!("`CoerceUnsized` implementation {}", err));
    });

    let source = tcx.type_of(impl_did);
    let trait_ref = tcx.impl_trait_ref(impl_did).unwrap();
    assert_eq!(trait_ref.def_id, coerce_unsized_trait);
    let target = trait_ref.substs.type_at(1);
    debug!("visit_implementation_of_coerce_unsized: {:?} -> {:?} (bound)", source, target);

    let param_env = tcx.param_env(impl_did);
    assert!(!source.has_escaping_bound_vars());

    let err_info = CoerceUnsizedInfo { custom_kind: None };

    debug!("visit_implementation_of_coerce_unsized: {:?} -> {:?} (free)", source, target);

    tcx.infer_ctxt().enter(|infcx| {
        let impl_hir_id = tcx.hir().local_def_id_to_hir_id(impl_did);
        let cause = ObligationCause::misc(span, impl_hir_id);
        let check_mutbl = |mt_a: ty::TypeAndMut<'tcx>,
                           mt_b: ty::TypeAndMut<'tcx>,
                           mk_ptr: &dyn Fn(Ty<'tcx>) -> Ty<'tcx>| {
            if (mt_a.mutbl, mt_b.mutbl) == (hir::Mutability::Not, hir::Mutability::Mut) {
                infcx
                    .report_mismatched_types(
                        &cause,
                        mk_ptr(mt_b.ty),
                        target,
                        ty::error::TypeError::Mutability,
                    )
                    .emit();
            }
            (mt_a.ty, mt_b.ty, unsize_trait, None)
        };
        let (source, target, trait_def_id, kind) = match (source.kind(), target.kind()) {
            (&ty::Ref(r_a, ty_a, mutbl_a), &ty::Ref(r_b, ty_b, mutbl_b)) => {
                infcx.sub_regions(infer::RelateObjectBound(span), r_b, r_a);
                let mt_a = ty::TypeAndMut { ty: ty_a, mutbl: mutbl_a };
                let mt_b = ty::TypeAndMut { ty: ty_b, mutbl: mutbl_b };
                check_mutbl(mt_a, mt_b, &|ty| tcx.mk_imm_ref(r_b, ty))
            }

            (&ty::Ref(_, ty_a, mutbl_a), &ty::RawPtr(mt_b)) => {
                let mt_a = ty::TypeAndMut { ty: ty_a, mutbl: mutbl_a };
                check_mutbl(mt_a, mt_b, &|ty| tcx.mk_imm_ptr(ty))
            }

            (&ty::RawPtr(mt_a), &ty::RawPtr(mt_b)) => {
                check_mutbl(mt_a, mt_b, &|ty| tcx.mk_imm_ptr(ty))
            }

            (&ty::Adt(def_a, substs_a), &ty::Adt(def_b, substs_b))
                if def_a.is_struct() && def_b.is_struct() =>
            {
                if def_a != def_b {
                    let source_path = tcx.def_path_str(def_a.did);
                    let target_path = tcx.def_path_str(def_b.did);
                    struct_span_err!(
                        tcx.sess,
                        span,
                        E0377,
                        "the trait `CoerceUnsized` may only be implemented \
                               for a coercion between structures with the same \
                               definition; expected `{}`, found `{}`",
                        source_path,
                        target_path
                    )
                    .emit();
                    return err_info;
                }

                // Here we are considering a case of converting
                // `S<P0...Pn>` to S<Q0...Qn>`. As an example, let's imagine a struct `Foo<T, U>`,
                // which acts like a pointer to `U`, but carries along some extra data of type `T`:
                //
                //     struct Foo<T, U> {
                //         extra: T,
                //         ptr: *mut U,
                //     }
                //
                // We might have an impl that allows (e.g.) `Foo<T, [i32; 3]>` to be unsized
                // to `Foo<T, [i32]>`. That impl would look like:
                //
                //   impl<T, U: Unsize<V>, V> CoerceUnsized<Foo<T, V>> for Foo<T, U> {}
                //
                // Here `U = [i32; 3]` and `V = [i32]`. At runtime,
                // when this coercion occurs, we would be changing the
                // field `ptr` from a thin pointer of type `*mut [i32;
                // 3]` to a fat pointer of type `*mut [i32]` (with
                // extra data `3`).  **The purpose of this check is to
                // make sure that we know how to do this conversion.**
                //
                // To check if this impl is legal, we would walk down
                // the fields of `Foo` and consider their types with
                // both substitutes. We are looking to find that
                // exactly one (non-phantom) field has changed its
                // type, which we will expect to be the pointer that
                // is becoming fat (we could probably generalize this
                // to multiple thin pointers of the same type becoming
                // fat, but we don't). In this case:
                //
                // - `extra` has type `T` before and type `T` after
                // - `ptr` has type `*mut U` before and type `*mut V` after
                //
                // Since just one field changed, we would then check
                // that `*mut U: CoerceUnsized<*mut V>` is implemented
                // (in other words, that we know how to do this
                // conversion). This will work out because `U:
                // Unsize<V>`, and we have a builtin rule that `*mut
                // U` can be coerced to `*mut V` if `U: Unsize<V>`.
                let fields = &def_a.non_enum_variant().fields;
                let diff_fields = fields
                    .iter()
                    .enumerate()
                    .filter_map(|(i, f)| {
                        let (a, b) = (f.ty(tcx, substs_a), f.ty(tcx, substs_b));

                        if tcx.type_of(f.did).is_phantom_data() {
                            // Ignore PhantomData fields
                            return None;
                        }

                        // Ignore fields that aren't changed; it may
                        // be that we could get away with subtyping or
                        // something more accepting, but we use
                        // equality because we want to be able to
                        // perform this check without computing
                        // variance where possible. (This is because
                        // we may have to evaluate constraint
                        // expressions in the course of execution.)
                        // See e.g., #41936.
                        if let Ok(ok) = infcx.at(&cause, param_env).eq(a, b) {
                            if ok.obligations.is_empty() {
                                return None;
                            }
                        }

                        // Collect up all fields that were significantly changed
                        // i.e., those that contain T in coerce_unsized T -> U
                        Some((i, a, b))
                    })
                    .collect::<Vec<_>>();

                if diff_fields.is_empty() {
                    struct_span_err!(
                        tcx.sess,
                        span,
                        E0374,
                        "the trait `CoerceUnsized` may only be implemented \
                               for a coercion between structures with one field \
                               being coerced, none found"
                    )
                    .emit();
                    return err_info;
                } else if diff_fields.len() > 1 {
                    let item = tcx.hir().expect_item(impl_did);
                    let span = if let ItemKind::Impl(hir::Impl { of_trait: Some(ref t), .. }) =
                        item.kind
                    {
                        t.path.span
                    } else {
                        tcx.def_span(impl_did)
                    };

                    struct_span_err!(
                        tcx.sess,
                        span,
                        E0375,
                        "implementing the trait \
                                                    `CoerceUnsized` requires multiple \
                                                    coercions"
                    )
                    .note(
                        "`CoerceUnsized` may only be implemented for \
                              a coercion between structures with one field being coerced",
                    )
                    .note(&format!(
                        "currently, {} fields need coercions: {}",
                        diff_fields.len(),
                        diff_fields
                            .iter()
                            .map(|&(i, a, b)| {
                                format!("`{}` (`{}` to `{}`)", fields[i].ident, a, b)
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                    .span_label(span, "requires multiple coercions")
                    .emit();
                    return err_info;
                }

                let (i, a, b) = diff_fields[0];
                let kind = ty::adjustment::CustomCoerceUnsized::Struct(i);
                (a, b, coerce_unsized_trait, Some(kind))
            }

            _ => {
                struct_span_err!(
                    tcx.sess,
                    span,
                    E0376,
                    "the trait `CoerceUnsized` may only be implemented \
                           for a coercion between structures"
                )
                .emit();
                return err_info;
            }
        };

        let mut fulfill_cx = <dyn TraitEngine<'_>>::new(infcx.tcx);

        // Register an obligation for `A: Trait<B>`.
        let cause = traits::ObligationCause::misc(span, impl_hir_id);
        let predicate = predicate_for_trait_def(
            tcx,
            param_env,
            cause,
            trait_def_id,
            0,
            source,
            &[target.into()],
        );
        fulfill_cx.register_predicate_obligation(&infcx, predicate);

        // Check that all transitive obligations are satisfied.
        let errors = fulfill_cx.select_all_or_error(&infcx);
        if !errors.is_empty() {
            infcx.report_fulfillment_errors(&errors, None, false);
        }

        // Finally, resolve all regions.
        let outlives_env = OutlivesEnvironment::new(param_env);
        infcx.resolve_regions_and_report_errors(
            impl_did.to_def_id(),
            &outlives_env,
            RegionckMode::default(),
        );

        CoerceUnsizedInfo { custom_kind: kind }
    })
}
