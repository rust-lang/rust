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

use rustc::middle::free_region::FreeRegionMap;
use rustc::middle::lang_items::UnsizeTraitLangItem;

use rustc::traits::{self, ObligationCause, Reveal};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::ParameterEnvironment;
use rustc::ty::TypeFoldable;
use rustc::ty::subst::Subst;
use rustc::ty::util::CopyImplementationError;
use rustc::infer;

use rustc::hir::def_id::DefId;
use rustc::hir::map as hir_map;
use rustc::hir::{self, ItemImpl};

pub fn check<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    check_trait(tcx, tcx.lang_items.drop_trait(), visit_implementation_of_drop);
    check_trait(tcx, tcx.lang_items.copy_trait(), visit_implementation_of_copy);
    check_trait(
        tcx,
        tcx.lang_items.coerce_unsized_trait(),
        visit_implementation_of_coerce_unsized);
}

fn check_trait<'a, 'tcx, F>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            trait_def_id: Option<DefId>,
                            mut f: F)
    where F: FnMut(TyCtxt<'a, 'tcx, 'tcx>, DefId, DefId)
{
    if let Some(trait_def_id) = trait_def_id {
        let mut impls = vec![];
        tcx.lookup_trait_def(trait_def_id).for_each_impl(tcx, |did| {
            impls.push(did);
        });
        impls.sort();
        for impl_def_id in impls {
            f(tcx, trait_def_id, impl_def_id);
        }
    }
}

fn visit_implementation_of_drop<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                          _drop_did: DefId,
                                          impl_did: DefId) {
    let items = tcx.associated_item_def_ids(impl_did);
    if items.is_empty() {
        // We'll error out later. For now, just don't ICE.
        return;
    }
    let method_def_id = items[0];

    let self_type = tcx.item_type(impl_did);
    match self_type.sty {
        ty::TyAdt(type_def, _) => {
            type_def.set_destructor(method_def_id);
        }
        _ => {
            // Destructors only work on nominal types.
            if let Some(impl_node_id) = tcx.map.as_local_node_id(impl_did) {
                match tcx.map.find(impl_node_id) {
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
                            .span_label(span, &format!("implementing Drop requires a struct"))
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

    let impl_node_id = if let Some(n) = tcx.map.as_local_node_id(impl_did) {
        n
    } else {
        debug!("visit_implementation_of_copy(): impl not in this \
                crate");
        return;
    };

    let self_type = tcx.item_type(impl_did);
    debug!("visit_implementation_of_copy: self_type={:?} (bound)",
           self_type);

    let span = tcx.map.span(impl_node_id);
    let param_env = ParameterEnvironment::for_item(tcx, impl_node_id);
    let self_type = self_type.subst(tcx, &param_env.free_substs);
    assert!(!self_type.has_escaping_regions());

    debug!("visit_implementation_of_copy: self_type={:?} (free)",
           self_type);

    match param_env.can_type_implement_copy(tcx, self_type, span) {
        Ok(()) => {}
        Err(CopyImplementationError::InfrigingField(field)) => {
            let item = tcx.map.expect_item(impl_node_id);
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
                    &"this field does not implement `Copy`")
                .emit()
        }
        Err(CopyImplementationError::NotAnAdt) => {
            let item = tcx.map.expect_item(impl_node_id);
            let span = if let ItemImpl(.., ref ty, _) = item.node {
                ty.span
            } else {
                span
            };

            struct_span_err!(tcx.sess,
                             span,
                             E0206,
                             "the trait `Copy` may not be implemented for this type")
                .span_label(span, &format!("type is not a structure or enumeration"))
                .emit();
        }
        Err(CopyImplementationError::HasDestructor) => {
            struct_span_err!(tcx.sess,
                             span,
                             E0184,
                             "the trait `Copy` may not be implemented for this type; the \
                              type has a destructor")
                .span_label(span, &format!("Copy not allowed on types with destructors"))
                .emit();
        }
    }
}

fn visit_implementation_of_coerce_unsized<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                    coerce_unsized_trait: DefId,
                                                    impl_did: DefId) {
    debug!("visit_implementation_of_coerce_unsized: impl_did={:?}",
           impl_did);

    let unsize_trait = match tcx.lang_items.require(UnsizeTraitLangItem) {
        Ok(id) => id,
        Err(err) => {
            tcx.sess.fatal(&format!("`CoerceUnsized` implementation {}", err));
        }
    };

    let impl_node_id = if let Some(n) = tcx.map.as_local_node_id(impl_did) {
        n
    } else {
        debug!("visit_implementation_of_coerce_unsized(): impl not \
                in this crate");
        return;
    };

    let source = tcx.item_type(impl_did);
    let trait_ref = tcx.impl_trait_ref(impl_did).unwrap();
    let target = trait_ref.substs.type_at(1);
    debug!("visit_implementation_of_coerce_unsized: {:?} -> {:?} (bound)",
           source,
           target);

    let span = tcx.map.span(impl_node_id);
    let param_env = ParameterEnvironment::for_item(tcx, impl_node_id);
    let source = source.subst(tcx, &param_env.free_substs);
    let target = target.subst(tcx, &param_env.free_substs);
    assert!(!source.has_escaping_regions());

    debug!("visit_implementation_of_coerce_unsized: {:?} -> {:?} (free)",
           source,
           target);

    tcx.infer_ctxt(param_env, Reveal::ExactMatch).enter(|infcx| {
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
            (&ty::TyBox(a), &ty::TyBox(b)) => (a, b, unsize_trait, None),

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
                    return;
                }

                let fields = &def_a.struct_variant().fields;
                let diff_fields = fields.iter()
                    .enumerate()
                    .filter_map(|(i, f)| {
                        let (a, b) = (f.ty(tcx, substs_a), f.ty(tcx, substs_b));

                        if tcx.item_type(f.did).is_phantom_data() {
                            // Ignore PhantomData fields
                            return None;
                        }

                        // Ignore fields that aren't significantly changed
                        if let Ok(ok) = infcx.sub_types(false, &cause, b, a) {
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
                    return;
                } else if diff_fields.len() > 1 {
                    let item = tcx.map.expect_item(impl_node_id);
                    let span = if let ItemImpl(.., Some(ref t), _, _) = item.node {
                        t.path.span
                    } else {
                        tcx.map.span(impl_node_id)
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
                    err.span_label(span, &format!("requires multiple coercions"));
                    err.emit();
                    return;
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
                return;
            }
        };

        let mut fulfill_cx = traits::FulfillmentContext::new();

        // Register an obligation for `A: Trait<B>`.
        let cause = traits::ObligationCause::misc(span, impl_node_id);
        let predicate = tcx.predicate_for_trait_def(cause, trait_def_id, 0, source, &[target]);
        fulfill_cx.register_predicate_obligation(&infcx, predicate);

        // Check that all transitive obligations are satisfied.
        if let Err(errors) = fulfill_cx.select_all_or_error(&infcx) {
            infcx.report_fulfillment_errors(&errors);
        }

        // Finally, resolve all regions.
        let mut free_regions = FreeRegionMap::new();
        free_regions.relate_free_regions_from_predicates(&infcx.parameter_environment
            .caller_bounds);
        infcx.resolve_regions_and_report_errors(&free_regions, impl_node_id);

        if let Some(kind) = kind {
            tcx.custom_coerce_unsized_kinds.borrow_mut().insert(impl_did, kind);
        }
    });
}
