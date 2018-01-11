// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Check properties that are required by built-in traits and set
//! up data structures required by type-checking/translation.

use rustc::infer::outlives::env::OutlivesEnvironment;
use rustc::middle::region;
use rustc::middle::lang_items::UnsizeTraitLangItem;

use rustc::traits::{self, ObligationCause};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::TypeFoldable;
use rustc::ty::adjustment::CoerceUnsizedInfo;
use rustc::ty::util::CopyImplementationError;
use rustc::infer;

use rustc::hir::def_id::DefId;
use rustc::hir::map as hir_map;
use rustc::hir::{self, ItemImpl};

pub fn check_trait<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, trait_def_id: DefId) {
    Checker { tcx, trait_def_id }
        .check(tcx.lang_items().drop_trait(), visit_implementation_of_drop)
        .check(tcx.lang_items().copy_trait(), visit_implementation_of_copy)
        .check(tcx.lang_items().coerce_unsized_trait(),
               visit_implementation_of_coerce_unsized);
}

struct Checker<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    trait_def_id: DefId
}

impl<'a, 'tcx> Checker<'a, 'tcx> {
    fn check<F>(&self, trait_def_id: Option<DefId>, mut f: F) -> &Self
        where F: FnMut(TyCtxt<'a, 'tcx, 'tcx>, DefId, DefId)
    {
        if Some(self.trait_def_id) == trait_def_id {
            for &impl_id in self.tcx.hir.trait_impls(self.trait_def_id) {
                let impl_def_id = self.tcx.hir.local_def_id(impl_id);
                f(self.tcx, self.trait_def_id, impl_def_id);
            }
        }
        self
    }
}

fn visit_implementation_of_drop<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                          _drop_did: DefId,
                                          impl_did: DefId) {
    match tcx.type_of(impl_did).sty {
        ty::TyAdt(..) => {}
        _ => {
            // Destructors only work on nominal types.
            if let Some(impl_node_id) = tcx.hir.as_local_node_id(impl_did) {
                match tcx.hir.find(impl_node_id) {
                    Some(hir_map::NodeItem(item)) => {
                        let span = match item.node {
                            ItemImpl(.., ref ty, _) => ty.span,
                            _ => item.span,
                        };
                        struct_span_err!(tcx.sess,
                                         span,
                                         E0120,
                                         "the Drop trait may only be implemented on \
                                         structures")
                            .span_label(span, "implementing Drop requires a struct")
                            .emit();
                    }
                    _ => {
                        bug!("didn't find impl in ast map");
                    }
                }
            } else {
                bug!("found external impl of Drop trait on \
                      something other than a struct");
            }
        }
    }
}

fn visit_implementation_of_copy<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                          _copy_did: DefId,
                                          impl_did: DefId) {
    debug!("visit_implementation_of_copy: impl_did={:?}", impl_did);

    let impl_node_id = if let Some(n) = tcx.hir.as_local_node_id(impl_did) {
        n
    } else {
        debug!("visit_implementation_of_copy(): impl not in this \
                crate");
        return;
    };

    let self_type = tcx.type_of(impl_did);
    debug!("visit_implementation_of_copy: self_type={:?} (bound)",
           self_type);

    let span = tcx.hir.span(impl_node_id);
    let param_env = tcx.param_env(impl_did);
    assert!(!self_type.has_escaping_regions());

    debug!("visit_implementation_of_copy: self_type={:?} (free)",
           self_type);

    match param_env.can_type_implement_copy(tcx, self_type, span) {
        Ok(()) => {}
        Err(CopyImplementationError::InfrigingField(field)) => {
            let item = tcx.hir.expect_item(impl_node_id);
            let span = if let ItemImpl(.., Some(ref tr), _, _) = item.node {
                tr.path.span
            } else {
                span
            };

            struct_span_err!(tcx.sess,
                             span,
                             E0204,
                             "the trait `Copy` may not be implemented for this type")
                .span_label(
                    tcx.def_span(field.did),
                    "this field does not implement `Copy`")
                .emit()
        }
        Err(CopyImplementationError::NotAnAdt) => {
            let item = tcx.hir.expect_item(impl_node_id);
            let span = if let ItemImpl(.., ref ty, _) = item.node {
                ty.span
            } else {
                span
            };

            struct_span_err!(tcx.sess,
                             span,
                             E0206,
                             "the trait `Copy` may not be implemented for this type")
                .span_label(span, "type is not a structure or enumeration")
                .emit();
        }
        Err(CopyImplementationError::HasDestructor) => {
            struct_span_err!(tcx.sess,
                             span,
                             E0184,
                             "the trait `Copy` may not be implemented for this type; the \
                              type has a destructor")
                .span_label(span, "Copy not allowed on types with destructors")
                .emit();
        }
    }
}

fn visit_implementation_of_coerce_unsized<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                    _: DefId,
                                                    impl_did: DefId) {
    debug!("visit_implementation_of_coerce_unsized: impl_did={:?}",
           impl_did);

    // Just compute this for the side-effects, in particular reporting
    // errors; other parts of the code may demand it for the info of
    // course.
    if impl_did.is_local() {
        let span = tcx.def_span(impl_did);
        tcx.at(span).coerce_unsized_info(impl_did);
    }
}

pub fn coerce_unsized_info<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     impl_did: DefId)
                                     -> CoerceUnsizedInfo {
    debug!("compute_coerce_unsized_info(impl_did={:?})", impl_did);
    let coerce_unsized_trait = tcx.lang_items().coerce_unsized_trait().unwrap();

    let unsize_trait = match tcx.lang_items().require(UnsizeTraitLangItem) {
        Ok(id) => id,
        Err(err) => {
            tcx.sess.fatal(&format!("`CoerceUnsized` implementation {}", err));
        }
    };

    // this provider should only get invoked for local def-ids
    let impl_node_id = tcx.hir.as_local_node_id(impl_did).unwrap_or_else(|| {
        bug!("coerce_unsized_info: invoked for non-local def-id {:?}", impl_did)
    });

    let source = tcx.type_of(impl_did);
    let trait_ref = tcx.impl_trait_ref(impl_did).unwrap();
    assert_eq!(trait_ref.def_id, coerce_unsized_trait);
    let target = trait_ref.substs.type_at(1);
    debug!("visit_implementation_of_coerce_unsized: {:?} -> {:?} (bound)",
           source,
           target);

    let span = tcx.hir.span(impl_node_id);
    let param_env = tcx.param_env(impl_did);
    assert!(!source.has_escaping_regions());

    let err_info = CoerceUnsizedInfo { custom_kind: None };

    debug!("visit_implementation_of_coerce_unsized: {:?} -> {:?} (free)",
           source,
           target);

    tcx.infer_ctxt().enter(|infcx| {
        let cause = ObligationCause::misc(span, impl_node_id);
        let check_mutbl = |mt_a: ty::TypeAndMut<'tcx>,
                           mt_b: ty::TypeAndMut<'tcx>,
                           mk_ptr: &Fn(Ty<'tcx>) -> Ty<'tcx>| {
            if (mt_a.mutbl, mt_b.mutbl) == (hir::MutImmutable, hir::MutMutable) {
                infcx.report_mismatched_types(&cause,
                                             mk_ptr(mt_b.ty),
                                             target,
                                             ty::error::TypeError::Mutability)
                    .emit();
            }
            (mt_a.ty, mt_b.ty, unsize_trait, None)
        };
        let (source, target, trait_def_id, kind) = match (&source.sty, &target.sty) {
            (&ty::TyRef(r_a, mt_a), &ty::TyRef(r_b, mt_b)) => {
                infcx.sub_regions(infer::RelateObjectBound(span), r_b, r_a);
                check_mutbl(mt_a, mt_b, &|ty| tcx.mk_imm_ref(r_b, ty))
            }

            (&ty::TyRef(_, mt_a), &ty::TyRawPtr(mt_b)) |
            (&ty::TyRawPtr(mt_a), &ty::TyRawPtr(mt_b)) => {
                check_mutbl(mt_a, mt_b, &|ty| tcx.mk_imm_ptr(ty))
            }

            (&ty::TyAdt(def_a, substs_a), &ty::TyAdt(def_b, substs_b)) if def_a.is_struct() &&
                                                                          def_b.is_struct() => {
                if def_a != def_b {
                    let source_path = tcx.item_path_str(def_a.did);
                    let target_path = tcx.item_path_str(def_b.did);
                    span_err!(tcx.sess,
                              span,
                              E0377,
                              "the trait `CoerceUnsized` may only be implemented \
                               for a coercion between structures with the same \
                               definition; expected {}, found {}",
                              source_path,
                              target_path);
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
                // to mutiple thin pointers of the same type becoming
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
                let diff_fields = fields.iter()
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
                        // See e.g. #41936.
                        if let Ok(ok) = infcx.at(&cause, param_env).eq(a, b) {
                            if ok.obligations.is_empty() {
                                return None;
                            }
                        }

                        // Collect up all fields that were significantly changed
                        // i.e. those that contain T in coerce_unsized T -> U
                        Some((i, a, b))
                    })
                    .collect::<Vec<_>>();

                if diff_fields.is_empty() {
                    span_err!(tcx.sess,
                              span,
                              E0374,
                              "the trait `CoerceUnsized` may only be implemented \
                               for a coercion between structures with one field \
                               being coerced, none found");
                    return err_info;
                } else if diff_fields.len() > 1 {
                    let item = tcx.hir.expect_item(impl_node_id);
                    let span = if let ItemImpl(.., Some(ref t), _, _) = item.node {
                        t.path.span
                    } else {
                        tcx.hir.span(impl_node_id)
                    };

                    let mut err = struct_span_err!(tcx.sess,
                                                   span,
                                                   E0375,
                                                   "implementing the trait \
                                                    `CoerceUnsized` requires multiple \
                                                    coercions");
                    err.note("`CoerceUnsized` may only be implemented for \
                              a coercion between structures with one field being coerced");
                    err.note(&format!("currently, {} fields need coercions: {}",
                                      diff_fields.len(),
                                      diff_fields.iter()
                                          .map(|&(i, a, b)| {
                                              format!("{} ({} to {})", fields[i].name, a, b)
                                          })
                                          .collect::<Vec<_>>()
                                          .join(", ")));
                    err.span_label(span, "requires multiple coercions");
                    err.emit();
                    return err_info;
                }

                let (i, a, b) = diff_fields[0];
                let kind = ty::adjustment::CustomCoerceUnsized::Struct(i);
                (a, b, coerce_unsized_trait, Some(kind))
            }

            _ => {
                span_err!(tcx.sess,
                          span,
                          E0376,
                          "the trait `CoerceUnsized` may only be implemented \
                           for a coercion between structures");
                return err_info;
            }
        };

        let mut fulfill_cx = traits::FulfillmentContext::new();

        // Register an obligation for `A: Trait<B>`.
        let cause = traits::ObligationCause::misc(span, impl_node_id);
        let predicate = tcx.predicate_for_trait_def(param_env,
                                                    cause,
                                                    trait_def_id,
                                                    0,
                                                    source,
                                                    &[target]);
        fulfill_cx.register_predicate_obligation(&infcx, predicate);

        // Check that all transitive obligations are satisfied.
        if let Err(errors) = fulfill_cx.select_all_or_error(&infcx) {
            infcx.report_fulfillment_errors(&errors, None);
        }

        // Finally, resolve all regions.
        let region_scope_tree = region::ScopeTree::default();
        let outlives_env = OutlivesEnvironment::new(param_env);
        infcx.resolve_regions_and_report_errors(
            impl_did,
            &region_scope_tree,
            &outlives_env,
        );

        CoerceUnsizedInfo {
            custom_kind: kind
        }
    })
}
