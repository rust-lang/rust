// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Coherence phase
//
// The job of the coherence phase of typechecking is to ensure that
// each trait has at most one implementation for each type. This is
// done by the orphan and overlap modules. Then we build up various
// mappings. That mapping code resides here.


use middle::def_id::DefId;
use middle::lang_items::UnsizeTraitLangItem;
use middle::subst::{self, Subst};
use middle::traits;
use middle::ty;
use middle::ty::RegionEscape;
use middle::ty::{ImplOrTraitItemId, ConstTraitItemId};
use middle::ty::{MethodTraitItemId, TypeTraitItemId, ParameterEnvironment};
use middle::ty::{Ty, TyBool, TyChar, TyEnum, TyError};
use middle::ty::{TyParam, TyRawPtr};
use middle::ty::{TyRef, TyStruct, TyTrait, TyTuple};
use middle::ty::{TyStr, TyArray, TySlice, TyFloat, TyInfer, TyInt};
use middle::ty::{TyUint, TyClosure, TyBox, TyBareFn};
use middle::ty::TyProjection;
use middle::ty::util::CopyImplementationError;
use middle::free_region::FreeRegionMap;
use CrateCtxt;
use middle::infer::{self, InferCtxt, new_infer_ctxt};
use std::cell::RefCell;
use std::rc::Rc;
use syntax::codemap::Span;
use syntax::parse::token;
use util::nodemap::{DefIdMap, FnvHashMap};
use rustc::front::map as hir_map;
use rustc::front::map::NodeItem;
use rustc_front::visit;
use rustc_front::hir::{Item, ItemImpl,Crate};
use rustc_front::hir;

mod orphan;
mod overlap;
mod unsafety;

// Returns the def ID of the base type, if there is one.
fn get_base_type_def_id<'a, 'tcx>(inference_context: &InferCtxt<'a, 'tcx>,
                                  span: Span,
                                  ty: Ty<'tcx>)
                                  -> Option<DefId> {
    match ty.sty {
        TyEnum(def, _) |
        TyStruct(def, _) => {
            Some(def.did)
        }

        TyTrait(ref t) => {
            Some(t.principal_def_id())
        }

        TyBox(_) => {
            inference_context.tcx.lang_items.owned_box()
        }

        TyBool | TyChar | TyInt(..) | TyUint(..) | TyFloat(..) |
        TyStr(..) | TyArray(..) | TySlice(..) | TyBareFn(..) | TyTuple(..) |
        TyParam(..) | TyError |
        TyRawPtr(_) | TyRef(_, _) | TyProjection(..) => {
            None
        }

        TyInfer(..) | TyClosure(..) => {
            // `ty` comes from a user declaration so we should only expect types
            // that the user can type
            inference_context.tcx.sess.span_bug(
                span,
                &format!("coherence encountered unexpected type searching for base type: {}",
                         ty));
        }
    }
}

struct CoherenceChecker<'a, 'tcx: 'a> {
    crate_context: &'a CrateCtxt<'a, 'tcx>,
    inference_context: InferCtxt<'a, 'tcx>,
    inherent_impls: RefCell<DefIdMap<Rc<RefCell<Vec<DefId>>>>>,
}

struct CoherenceCheckVisitor<'a, 'tcx: 'a> {
    cc: &'a CoherenceChecker<'a, 'tcx>
}

impl<'a, 'tcx, 'v> visit::Visitor<'v> for CoherenceCheckVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &Item) {
        if let ItemImpl(..) = item.node {
            self.cc.check_implementation(item)
        }

        visit::walk_item(self, item);
    }
}

impl<'a, 'tcx> CoherenceChecker<'a, 'tcx> {
    fn check(&self, krate: &Crate) {
        // Check implementations and traits. This populates the tables
        // containing the inherent methods and extension methods. It also
        // builds up the trait inheritance table.
        let mut visitor = CoherenceCheckVisitor { cc: self };
        visit::walk_crate(&mut visitor, krate);

        // Copy over the inherent impls we gathered up during the walk into
        // the tcx.
        let mut tcx_inherent_impls =
            self.crate_context.tcx.inherent_impls.borrow_mut();
        for (k, v) in self.inherent_impls.borrow().iter() {
            tcx_inherent_impls.insert((*k).clone(),
                                      Rc::new((*v.borrow()).clone()));
        }

        // Populate the table of destructors. It might seem a bit strange to
        // do this here, but it's actually the most convenient place, since
        // the coherence tables contain the trait -> type mappings.
        self.populate_destructors();

        // Check to make sure implementations of `Copy` are legal.
        self.check_implementations_of_copy();

        // Check to make sure implementations of `CoerceUnsized` are legal
        // and collect the necessary information from them.
        self.check_implementations_of_coerce_unsized();
    }

    fn check_implementation(&self, item: &Item) {
        let tcx = self.crate_context.tcx;
        let impl_did = tcx.map.local_def_id(item.id);
        let self_type = tcx.lookup_item_type(impl_did);

        // If there are no traits, then this implementation must have a
        // base type.

        let impl_items = self.create_impl_from_item(item);

        if let Some(trait_ref) = self.crate_context.tcx.impl_trait_ref(impl_did) {
            debug!("(checking implementation) adding impl for trait '{:?}', item '{}'",
                   trait_ref,
                   item.name);

            enforce_trait_manually_implementable(self.crate_context.tcx,
                                                 item.span,
                                                 trait_ref.def_id);
            self.add_trait_impl(trait_ref, impl_did);
        } else {
            // Add the implementation to the mapping from implementation to base
            // type def ID, if there is a base type for this implementation and
            // the implementation does not have any associated traits.
            if let Some(base_type_def_id) = get_base_type_def_id(
                    &self.inference_context, item.span, self_type.ty) {
                self.add_inherent_impl(base_type_def_id, impl_did);
            }
        }

        tcx.impl_items.borrow_mut().insert(impl_did, impl_items);
    }

    fn add_inherent_impl(&self, base_def_id: DefId, impl_def_id: DefId) {
        match self.inherent_impls.borrow().get(&base_def_id) {
            Some(implementation_list) => {
                implementation_list.borrow_mut().push(impl_def_id);
                return;
            }
            None => {}
        }

        self.inherent_impls.borrow_mut().insert(
            base_def_id,
            Rc::new(RefCell::new(vec!(impl_def_id))));
    }

    fn add_trait_impl(&self, impl_trait_ref: ty::TraitRef<'tcx>, impl_def_id: DefId) {
        debug!("add_trait_impl: impl_trait_ref={:?} impl_def_id={:?}",
               impl_trait_ref, impl_def_id);
        let trait_def = self.crate_context.tcx.lookup_trait_def(impl_trait_ref.def_id);
        trait_def.record_impl(self.crate_context.tcx, impl_def_id, impl_trait_ref);
    }

    // Converts an implementation in the AST to a vector of items.
    fn create_impl_from_item(&self, item: &Item) -> Vec<ImplOrTraitItemId> {
        match item.node {
            ItemImpl(_, _, _, _, _, ref impl_items) => {
                impl_items.iter().map(|impl_item| {
                    let impl_def_id = self.crate_context.tcx.map.local_def_id(impl_item.id);
                    match impl_item.node {
                        hir::ImplItem_::Const(..) => {
                            ConstTraitItemId(impl_def_id)
                        }
                        hir::ImplItem_::Method(..) => {
                            MethodTraitItemId(impl_def_id)
                        }
                        hir::ImplItem_::Type(_) => {
                            TypeTraitItemId(impl_def_id)
                        }
                    }
                }).collect()
            }
            _ => {
                self.crate_context.tcx.sess.span_bug(item.span,
                                                     "can't convert a non-impl \
                                                      to an impl");
            }
        }
    }

    //
    // Destructors
    //

    fn populate_destructors(&self) {
        let tcx = self.crate_context.tcx;
        let drop_trait = match tcx.lang_items.drop_trait() {
            Some(id) => id, None => { return }
        };
        tcx.populate_implementations_for_trait_if_necessary(drop_trait);
        let drop_trait = tcx.lookup_trait_def(drop_trait);

        let impl_items = tcx.impl_items.borrow();

        drop_trait.for_each_impl(tcx, |impl_did| {
            let items = impl_items.get(&impl_did).unwrap();
            if items.is_empty() {
                // We'll error out later. For now, just don't ICE.
                return;
            }
            let method_def_id = items[0];

            let self_type = tcx.lookup_item_type(impl_did);
            match self_type.ty.sty {
                ty::TyEnum(type_def, _) |
                ty::TyStruct(type_def, _) => {
                    type_def.set_destructor(method_def_id.def_id());
                }
                _ => {
                    // Destructors only work on nominal types.
                    if let Some(impl_node_id) = tcx.map.as_local_node_id(impl_did) {
                        match tcx.map.find(impl_node_id) {
                            Some(hir_map::NodeItem(item)) => {
                                span_err!(tcx.sess, item.span, E0120,
                                          "the Drop trait may only be implemented on structures");
                            }
                            _ => {
                                tcx.sess.bug("didn't find impl in ast \
                                              map");
                            }
                        }
                    } else {
                        tcx.sess.bug("found external impl of Drop trait on \
                                      something other than a struct");
                    }
                }
            }
        });
    }

    /// Ensures that implementations of the built-in trait `Copy` are legal.
    fn check_implementations_of_copy(&self) {
        let tcx = self.crate_context.tcx;
        let copy_trait = match tcx.lang_items.copy_trait() {
            Some(id) => id,
            None => return,
        };
        tcx.populate_implementations_for_trait_if_necessary(copy_trait);
        let copy_trait = tcx.lookup_trait_def(copy_trait);

        copy_trait.for_each_impl(tcx, |impl_did| {
            debug!("check_implementations_of_copy: impl_did={:?}",
                   impl_did);

            let impl_node_id = if let Some(n) = tcx.map.as_local_node_id(impl_did) {
                n
            } else {
                debug!("check_implementations_of_copy(): impl not in this \
                        crate");
                return
            };

            let self_type = tcx.lookup_item_type(impl_did);
            debug!("check_implementations_of_copy: self_type={:?} (bound)",
                   self_type);

            let span = tcx.map.span(impl_node_id);
            let param_env = ParameterEnvironment::for_item(tcx, impl_node_id);
            let self_type = self_type.ty.subst(tcx, &param_env.free_substs);
            assert!(!self_type.has_escaping_regions());

            debug!("check_implementations_of_copy: self_type={:?} (free)",
                   self_type);

            match param_env.can_type_implement_copy(self_type, span) {
                Ok(()) => {}
                Err(CopyImplementationError::InfrigingField(name)) => {
                       span_err!(tcx.sess, span, E0204,
                                 "the trait `Copy` may not be \
                                          implemented for this type; field \
                                          `{}` does not implement `Copy`",
                                         name)
                }
                Err(CopyImplementationError::InfrigingVariant(name)) => {
                       span_err!(tcx.sess, span, E0205,
                                 "the trait `Copy` may not be \
                                          implemented for this type; variant \
                                          `{}` does not implement `Copy`",
                                         name)
                }
                Err(CopyImplementationError::NotAnAdt) => {
                       span_err!(tcx.sess, span, E0206,
                                 "the trait `Copy` may not be implemented \
                                  for this type; type is not a structure or \
                                  enumeration")
                }
                Err(CopyImplementationError::HasDestructor) => {
                    span_err!(tcx.sess, span, E0184,
                              "the trait `Copy` may not be implemented for this type; \
                               the type has a destructor");
                }
            }
        });
    }

    /// Process implementations of the built-in trait `CoerceUnsized`.
    fn check_implementations_of_coerce_unsized(&self) {
        let tcx = self.crate_context.tcx;
        let coerce_unsized_trait = match tcx.lang_items.coerce_unsized_trait() {
            Some(id) => id,
            None => return,
        };
        let unsize_trait = match tcx.lang_items.require(UnsizeTraitLangItem) {
            Ok(id) => id,
            Err(err) => {
                tcx.sess.fatal(&format!("`CoerceUnsized` implementation {}", err));
            }
        };

        let trait_def = tcx.lookup_trait_def(coerce_unsized_trait);

        trait_def.for_each_impl(tcx, |impl_did| {
            debug!("check_implementations_of_coerce_unsized: impl_did={:?}",
                   impl_did);

            let impl_node_id = if let Some(n) = tcx.map.as_local_node_id(impl_did) {
                n
            } else {
                debug!("check_implementations_of_coerce_unsized(): impl not \
                        in this crate");
                return;
            };

            let source = tcx.lookup_item_type(impl_did).ty;
            let trait_ref = self.crate_context.tcx.impl_trait_ref(impl_did).unwrap();
            let target = *trait_ref.substs.types.get(subst::TypeSpace, 0);
            debug!("check_implementations_of_coerce_unsized: {:?} -> {:?} (bound)",
                   source, target);

            let span = tcx.map.span(impl_node_id);
            let param_env = ParameterEnvironment::for_item(tcx, impl_node_id);
            let source = source.subst(tcx, &param_env.free_substs);
            let target = target.subst(tcx, &param_env.free_substs);
            assert!(!source.has_escaping_regions());

            debug!("check_implementations_of_coerce_unsized: {:?} -> {:?} (free)",
                   source, target);

            let infcx = new_infer_ctxt(tcx, &tcx.tables, Some(param_env), true);

            let check_mutbl = |mt_a: ty::TypeAndMut<'tcx>, mt_b: ty::TypeAndMut<'tcx>,
                               mk_ptr: &Fn(Ty<'tcx>) -> Ty<'tcx>| {
                if (mt_a.mutbl, mt_b.mutbl) == (hir::MutImmutable, hir::MutMutable) {
                    infcx.report_mismatched_types(span, mk_ptr(mt_b.ty),
                                                  target, &ty::error::TypeError::Mutability);
                }
                (mt_a.ty, mt_b.ty, unsize_trait, None)
            };
            let (source, target, trait_def_id, kind) = match (&source.sty, &target.sty) {
                (&ty::TyBox(a), &ty::TyBox(b)) => (a, b, unsize_trait, None),

                (&ty::TyRef(r_a, mt_a), &ty::TyRef(r_b, mt_b)) => {
                    infer::mk_subr(&infcx, infer::RelateObjectBound(span), *r_b, *r_a);
                    check_mutbl(mt_a, mt_b, &|ty| tcx.mk_imm_ref(r_b, ty))
                }

                (&ty::TyRef(_, mt_a), &ty::TyRawPtr(mt_b)) |
                (&ty::TyRawPtr(mt_a), &ty::TyRawPtr(mt_b)) => {
                    check_mutbl(mt_a, mt_b, &|ty| tcx.mk_imm_ptr(ty))
                }

                (&ty::TyStruct(def_a, substs_a), &ty::TyStruct(def_b, substs_b)) => {
                    if def_a != def_b {
                        let source_path = tcx.item_path_str(def_a.did);
                        let target_path = tcx.item_path_str(def_b.did);
                        span_err!(tcx.sess, span, E0377,
                                  "the trait `CoerceUnsized` may only be implemented \
                                   for a coercion between structures with the same \
                                   definition; expected {}, found {}",
                                  source_path, target_path);
                        return;
                    }

                    let origin = infer::Misc(span);
                    let fields = &def_a.struct_variant().fields;
                    let diff_fields = fields.iter().enumerate().filter_map(|(i, f)| {
                        let (a, b) = (f.ty(tcx, substs_a), f.ty(tcx, substs_b));

                        if f.unsubst_ty().is_phantom_data() {
                            // Ignore PhantomData fields
                            None
                        } else if infcx.sub_types(false, origin, b, a).is_ok() {
                            // Ignore fields that aren't significantly changed
                            None
                        } else {
                            // Collect up all fields that were significantly changed
                            // i.e. those that contain T in coerce_unsized T -> U
                            Some((i, a, b))
                        }
                    }).collect::<Vec<_>>();

                    if diff_fields.is_empty() {
                        span_err!(tcx.sess, span, E0374,
                                  "the trait `CoerceUnsized` may only be implemented \
                                   for a coercion between structures with one field \
                                   being coerced, none found");
                        return;
                    } else if diff_fields.len() > 1 {
                        span_err!(tcx.sess, span, E0375,
                                  "the trait `CoerceUnsized` may only be implemented \
                                   for a coercion between structures with one field \
                                   being coerced, but {} fields need coercions: {}",
                                   diff_fields.len(), diff_fields.iter().map(|&(i, a, b)| {
                                        let name = fields[i].name;
                                        format!("{} ({} to {})",
                                                if name == token::special_names::unnamed_field {
                                                    i.to_string()
                                                } else {
                                                    name.to_string()
                                                }, a, b)
                                   }).collect::<Vec<_>>().join(", "));
                        return;
                    }

                    let (i, a, b) = diff_fields[0];
                    let kind = ty::adjustment::CustomCoerceUnsized::Struct(i);
                    (a, b, coerce_unsized_trait, Some(kind))
                }

                _ => {
                    span_err!(tcx.sess, span, E0376,
                              "the trait `CoerceUnsized` may only be implemented \
                               for a coercion between structures");
                    return;
                }
            };

            let mut fulfill_cx = infcx.fulfillment_cx.borrow_mut();

            // Register an obligation for `A: Trait<B>`.
            let cause = traits::ObligationCause::misc(span, impl_node_id);
            let predicate = traits::predicate_for_trait_def(tcx, cause, trait_def_id,
                                                            0, source, vec![target]);
            fulfill_cx.register_predicate_obligation(&infcx, predicate);

            // Check that all transitive obligations are satisfied.
            if let Err(errors) = fulfill_cx.select_all_or_error(&infcx) {
                traits::report_fulfillment_errors(&infcx, &errors);
            }

            // Finally, resolve all regions.
            let mut free_regions = FreeRegionMap::new();
            free_regions.relate_free_regions_from_predicates(tcx, &infcx.parameter_environment
                                                                        .caller_bounds);
            infcx.resolve_regions_and_report_errors(&free_regions, impl_node_id);

            if let Some(kind) = kind {
                tcx.custom_coerce_unsized_kinds.borrow_mut().insert(impl_did, kind);
            }
        });
    }
}

fn enforce_trait_manually_implementable(tcx: &ty::ctxt, sp: Span, trait_def_id: DefId) {
    if tcx.sess.features.borrow().unboxed_closures {
        // the feature gate allows all of them
        return
    }
    let did = Some(trait_def_id);
    let li = &tcx.lang_items;

    let trait_name = if did == li.fn_trait() {
        "Fn"
    } else if did == li.fn_mut_trait() {
        "FnMut"
    } else if did == li.fn_once_trait() {
        "FnOnce"
    } else {
        return // everything OK
    };
    span_err!(tcx.sess, sp, E0183, "manual implementations of `{}` are experimental", trait_name);
    fileline_help!(tcx.sess, sp,
               "add `#![feature(unboxed_closures)]` to the crate attributes to enable");
}

pub fn check_coherence(crate_context: &CrateCtxt) {
    CoherenceChecker {
        crate_context: crate_context,
        inference_context: new_infer_ctxt(crate_context.tcx, &crate_context.tcx.tables, None, true),
        inherent_impls: RefCell::new(FnvHashMap()),
    }.check(crate_context.tcx.map.krate());
    unsafety::check(crate_context.tcx);
    orphan::check(crate_context.tcx);
    overlap::check(crate_context.tcx);
}
