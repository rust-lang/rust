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


use metadata::csearch::{each_impl, get_impl_trait};
use metadata::csearch;
use middle::subst;
use middle::subst::{Substs};
use middle::ty::get;
use middle::ty::{ImplContainer, ImplOrTraitItemId, MethodTraitItemId};
use middle::ty::{lookup_item_type};
use middle::ty::{t, ty_bool, ty_char, ty_bot, ty_box, ty_enum, ty_err};
use middle::ty::{ty_str, ty_vec, ty_float, ty_infer, ty_int, ty_nil, ty_open};
use middle::ty::{ty_param, Polytype, ty_ptr};
use middle::ty::{ty_rptr, ty_struct, ty_trait, ty_tup};
use middle::ty::{ty_uint, ty_unboxed_closure, ty_uniq, ty_bare_fn};
use middle::ty::{ty_closure};
use middle::ty::type_is_ty_var;
use middle::subst::Subst;
use middle::ty;
use middle::typeck::CrateCtxt;
use middle::typeck::infer::combine::Combine;
use middle::typeck::infer::InferCtxt;
use middle::typeck::infer::{new_infer_ctxt, resolve_ivar, resolve_type};
use std::collections::{HashSet};
use std::cell::RefCell;
use std::rc::Rc;
use syntax::ast::{Crate, DefId};
use syntax::ast::{Item, ItemImpl};
use syntax::ast::{LOCAL_CRATE, TraitRef};
use syntax::ast;
use syntax::ast_map::NodeItem;
use syntax::ast_map;
use syntax::ast_util::{local_def};
use syntax::codemap::{Span};
use syntax::parse::token;
use syntax::visit;
use util::nodemap::{DefIdMap, FnvHashMap};
use util::ppaux::Repr;

mod orphan;
mod overlap;

fn get_base_type(inference_context: &InferCtxt,
                 span: Span,
                 original_type: t)
                 -> Option<t> {
    let resolved_type = match resolve_type(inference_context,
                                           Some(span),
                                           original_type,
                                           resolve_ivar) {
        Ok(resulting_type) if !type_is_ty_var(resulting_type) => resulting_type,
        _ => {
            inference_context.tcx.sess.span_fatal(span,
                                                  "the type of this value must be known in order \
                                                   to determine the base type");
        }
    };

    match get(resolved_type).sty {
        ty_enum(..) | ty_struct(..) | ty_unboxed_closure(..) => {
            debug!("(getting base type) found base type");
            Some(resolved_type)
        }

        _ if ty::type_is_trait(resolved_type) => {
            debug!("(getting base type) found base type (trait)");
            Some(resolved_type)
        }

        ty_nil | ty_bot | ty_bool | ty_char | ty_int(..) | ty_uint(..) | ty_float(..) |
        ty_str(..) | ty_vec(..) | ty_bare_fn(..) | ty_closure(..) | ty_tup(..) |
        ty_infer(..) | ty_param(..) | ty_err | ty_open(..) |
        ty_box(_) | ty_uniq(_) | ty_ptr(_) | ty_rptr(_, _) => {
            debug!("(getting base type) no base type; found {:?}",
                   get(original_type).sty);
            None
        }
        ty_trait(..) => fail!("should have been caught")
    }
}

// Returns the def ID of the base type, if there is one.
fn get_base_type_def_id(inference_context: &InferCtxt,
                        span: Span,
                        original_type: t)
                        -> Option<DefId> {
    match get_base_type(inference_context, span, original_type) {
        None => None,
        Some(base_type) => {
            match get(base_type).sty {
                ty_enum(def_id, _) |
                ty_struct(def_id, _) |
                ty_unboxed_closure(def_id, _) => {
                    Some(def_id)
                }
                ty_ptr(ty::mt {ty, ..}) |
                ty_rptr(_, ty::mt {ty, ..}) |
                ty_uniq(ty) => {
                    match ty::get(ty).sty {
                        ty_trait(box ty::TyTrait { def_id, .. }) => {
                            Some(def_id)
                        }
                        _ => {
                            fail!("get_base_type() returned a type that wasn't an \
                                   enum, struct, or trait");
                        }
                    }
                }
                ty_trait(box ty::TyTrait { def_id, .. }) => {
                    Some(def_id)
                }
                _ => {
                    fail!("get_base_type() returned a type that wasn't an \
                           enum, struct, or trait");
                }
            }
        }
    }
}

struct CoherenceChecker<'a, 'tcx: 'a> {
    crate_context: &'a CrateCtxt<'a, 'tcx>,
    inference_context: InferCtxt<'a, 'tcx>,
    inherent_impls: RefCell<DefIdMap<Rc<RefCell<Vec<ast::DefId>>>>>,
}

struct CoherenceCheckVisitor<'a, 'tcx: 'a> {
    cc: &'a CoherenceChecker<'a, 'tcx>
}

impl<'a, 'tcx, 'v> visit::Visitor<'v> for CoherenceCheckVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &Item) {

        //debug!("(checking coherence) item '{}'", token::get_ident(item.ident));

        match item.node {
            ItemImpl(_, ref opt_trait, _, _) => {
                match opt_trait.clone() {
                    Some(opt_trait) => {
                        self.cc.check_implementation(item, [opt_trait]);
                    }
                    None => self.cc.check_implementation(item, [])
                }
            }
            _ => {
                // Nothing to do.
            }
        };

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

        // Bring in external crates. It's fine for this to happen after the
        // coherence checks, because we ensure by construction that no errors
        // can happen at link time.
        self.add_external_crates();

        // Populate the table of destructors. It might seem a bit strange to
        // do this here, but it's actually the most convenient place, since
        // the coherence tables contain the trait -> type mappings.
        self.populate_destructor_table();
    }

    fn check_implementation(&self,
                            item: &Item,
                            associated_traits: &[TraitRef]) {
        let tcx = self.crate_context.tcx;
        let impl_did = local_def(item.id);
        let self_type = ty::lookup_item_type(tcx, impl_did);

        // If there are no traits, then this implementation must have a
        // base type.

        let impl_items = self.create_impl_from_item(item);

        for associated_trait in associated_traits.iter() {
            let trait_ref = ty::node_id_to_trait_ref(
                self.crate_context.tcx, associated_trait.ref_id);
            debug!("(checking implementation) adding impl for trait '{}', item '{}'",
                   trait_ref.repr(self.crate_context.tcx),
                   token::get_ident(item.ident));

            self.add_trait_impl(trait_ref.def_id, impl_did);
        }

        // Add the implementation to the mapping from implementation to base
        // type def ID, if there is a base type for this implementation and
        // the implementation does not have any associated traits.
        match get_base_type_def_id(&self.inference_context,
                                   item.span,
                                   self_type.ty) {
            None => {
                // Nothing to do.
            }
            Some(base_type_def_id) => {
                // FIXME: Gather up default methods?
                if associated_traits.len() == 0 {
                    self.add_inherent_impl(base_type_def_id, impl_did);
                }
            }
        }

        tcx.impl_items.borrow_mut().insert(impl_did, impl_items);
    }

    // Creates default method IDs and performs type substitutions for an impl
    // and trait pair. Then, for each provided method in the trait, inserts a
    // `ProvidedMethodInfo` instance into the `provided_method_sources` map.
    fn instantiate_default_methods(
            &self,
            impl_id: DefId,
            trait_ref: &ty::TraitRef,
            all_impl_items: &mut Vec<ImplOrTraitItemId>) {
        let tcx = self.crate_context.tcx;
        debug!("instantiate_default_methods(impl_id={:?}, trait_ref={})",
               impl_id, trait_ref.repr(tcx));

        let impl_poly_type = ty::lookup_item_type(tcx, impl_id);

        let prov = ty::provided_trait_methods(tcx, trait_ref.def_id);
        for trait_method in prov.iter() {
            // Synthesize an ID.
            let new_id = tcx.sess.next_node_id();
            let new_did = local_def(new_id);

            debug!("new_did={:?} trait_method={}", new_did, trait_method.repr(tcx));

            // Create substitutions for the various trait parameters.
            let new_method_ty =
                Rc::new(subst_receiver_types_in_method_ty(
                    tcx,
                    impl_id,
                    &impl_poly_type,
                    trait_ref,
                    new_did,
                    &**trait_method,
                    Some(trait_method.def_id)));

            debug!("new_method_ty={}", new_method_ty.repr(tcx));
            all_impl_items.push(MethodTraitItemId(new_did));

            // construct the polytype for the method based on the
            // method_ty.  it will have all the generics from the
            // impl, plus its own.
            let new_polytype = ty::Polytype {
                generics: new_method_ty.generics.clone(),
                ty: ty::mk_bare_fn(tcx, new_method_ty.fty.clone())
            };
            debug!("new_polytype={}", new_polytype.repr(tcx));

            tcx.tcache.borrow_mut().insert(new_did, new_polytype);
            tcx.impl_or_trait_items
               .borrow_mut()
               .insert(new_did, ty::MethodTraitItem(new_method_ty));

            // Pair the new synthesized ID up with the
            // ID of the method.
            self.crate_context.tcx.provided_method_sources.borrow_mut()
                .insert(new_did, trait_method.def_id);
        }
    }

    fn add_inherent_impl(&self, base_def_id: DefId, impl_def_id: DefId) {
        match self.inherent_impls.borrow().find(&base_def_id) {
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

    fn add_trait_impl(&self, base_def_id: DefId, impl_def_id: DefId) {
        debug!("add_trait_impl: base_def_id={} impl_def_id={}",
               base_def_id, impl_def_id);
        ty::record_trait_implementation(self.crate_context.tcx,
                                        base_def_id,
                                        impl_def_id);
    }

    fn get_self_type_for_implementation(&self, impl_did: DefId)
                                        -> Polytype {
        self.crate_context.tcx.tcache.borrow().get_copy(&impl_did)
    }

    // Converts an implementation in the AST to a vector of items.
    fn create_impl_from_item(&self, item: &Item) -> Vec<ImplOrTraitItemId> {
        match item.node {
            ItemImpl(_, ref trait_refs, _, ref ast_items) => {
                let mut items: Vec<ImplOrTraitItemId> =
                        ast_items.iter()
                                 .map(|ast_item| {
                            match *ast_item {
                                ast::MethodImplItem(ref ast_method) => {
                                    MethodTraitItemId(
                                        local_def(ast_method.id))
                                }
                            }
                        }).collect();

                for trait_ref in trait_refs.iter() {
                    let ty_trait_ref = ty::node_id_to_trait_ref(
                        self.crate_context.tcx,
                        trait_ref.ref_id);

                    self.instantiate_default_methods(local_def(item.id),
                                                     &*ty_trait_ref,
                                                     &mut items);
                }

                items
            }
            _ => {
                self.crate_context.tcx.sess.span_bug(item.span,
                                                     "can't convert a non-impl to an impl");
            }
        }
    }

    // External crate handling

    fn add_external_impl(&self,
                         impls_seen: &mut HashSet<DefId>,
                         impl_def_id: DefId) {
        let tcx = self.crate_context.tcx;
        let impl_items = csearch::get_impl_items(&tcx.sess.cstore,
                                                 impl_def_id);

        // Make sure we don't visit the same implementation multiple times.
        if !impls_seen.insert(impl_def_id) {
            // Skip this one.
            return
        }
        // Good. Continue.

        let _ = lookup_item_type(tcx, impl_def_id);
        let associated_traits = get_impl_trait(tcx, impl_def_id);

        // Do a sanity check.
        assert!(associated_traits.is_some());

        // Record all the trait items.
        for trait_ref in associated_traits.iter() {
            self.add_trait_impl(trait_ref.def_id, impl_def_id);
        }

        // For any methods that use a default implementation, add them to
        // the map. This is a bit unfortunate.
        for item_def_id in impl_items.iter() {
            let impl_item = ty::impl_or_trait_item(tcx, item_def_id.def_id());
            match impl_item {
                ty::MethodTraitItem(ref method) => {
                    for &source in method.provided_source.iter() {
                        tcx.provided_method_sources
                           .borrow_mut()
                           .insert(item_def_id.def_id(), source);
                    }
                }
            }
        }

        tcx.impl_items.borrow_mut().insert(impl_def_id, impl_items);
    }

    // Adds implementations and traits from external crates to the coherence
    // info.
    fn add_external_crates(&self) {
        let mut impls_seen = HashSet::new();

        let crate_store = &self.crate_context.tcx.sess.cstore;
        crate_store.iter_crate_data(|crate_number, _crate_metadata| {
            each_impl(crate_store, crate_number, |def_id| {
                assert_eq!(crate_number, def_id.krate);
                self.add_external_impl(&mut impls_seen, def_id)
            })
        })
    }

    //
    // Destructors
    //

    fn populate_destructor_table(&self) {
        let tcx = self.crate_context.tcx;
        let drop_trait = match tcx.lang_items.drop_trait() {
            Some(id) => id, None => { return }
        };

        let impl_items = tcx.impl_items.borrow();
        let trait_impls = match tcx.trait_impls.borrow().find_copy(&drop_trait) {
            None => return, // No types with (new-style) dtors present.
            Some(found_impls) => found_impls
        };

        for &impl_did in trait_impls.borrow().iter() {
            let items = impl_items.get(&impl_did);
            if items.len() < 1 {
                // We'll error out later. For now, just don't ICE.
                continue;
            }
            let method_def_id = *items.get(0);

            let self_type = self.get_self_type_for_implementation(impl_did);
            match ty::get(self_type.ty).sty {
                ty::ty_enum(type_def_id, _) |
                ty::ty_struct(type_def_id, _) |
                ty::ty_unboxed_closure(type_def_id, _) => {
                    tcx.destructor_for_type
                       .borrow_mut()
                       .insert(type_def_id, method_def_id.def_id());
                    tcx.destructors
                       .borrow_mut()
                       .insert(method_def_id.def_id());
                }
                _ => {
                    // Destructors only work on nominal types.
                    if impl_did.krate == ast::LOCAL_CRATE {
                        {
                            match tcx.map.find(impl_did.node) {
                                Some(ast_map::NodeItem(item)) => {
                                    span_err!(tcx.sess, item.span, E0120,
                                        "the Drop trait may only be implemented on structures");
                                }
                                _ => {
                                    tcx.sess.bug("didn't find impl in ast \
                                                  map");
                                }
                            }
                        }
                    } else {
                        tcx.sess.bug("found external impl of Drop trait on \
                                      something other than a struct");
                    }
                }
            }
        }
    }
}

pub fn make_substs_for_receiver_types(tcx: &ty::ctxt,
                                      trait_ref: &ty::TraitRef,
                                      method: &ty::Method)
                                      -> subst::Substs
{
    /*!
     * Substitutes the values for the receiver's type parameters
     * that are found in method, leaving the method's type parameters
     * intact.
     */

    let meth_tps: Vec<ty::t> =
        method.generics.types.get_slice(subst::FnSpace)
              .iter()
              .map(|def| ty::mk_param_from_def(tcx, def))
              .collect();
    let meth_regions: Vec<ty::Region> =
        method.generics.regions.get_slice(subst::FnSpace)
              .iter()
              .map(|def| ty::ReEarlyBound(def.def_id.node, def.space,
                                          def.index, def.name))
              .collect();
    trait_ref.substs.clone().with_method(meth_tps, meth_regions)
}

fn subst_receiver_types_in_method_ty(tcx: &ty::ctxt,
                                     impl_id: ast::DefId,
                                     impl_poly_type: &ty::Polytype,
                                     trait_ref: &ty::TraitRef,
                                     new_def_id: ast::DefId,
                                     method: &ty::Method,
                                     provided_source: Option<ast::DefId>)
                                     -> ty::Method
{
    let combined_substs = make_substs_for_receiver_types(tcx, trait_ref, method);

    debug!("subst_receiver_types_in_method_ty: combined_substs={}",
           combined_substs.repr(tcx));

    let mut method_generics = method.generics.subst(tcx, &combined_substs);

    // replace the type parameters declared on the trait with those
    // from the impl
    for &space in [subst::TypeSpace, subst::SelfSpace].iter() {
        method_generics.types.replace(
            space,
            Vec::from_slice(impl_poly_type.generics.types.get_slice(space)));
        method_generics.regions.replace(
            space,
            Vec::from_slice(impl_poly_type.generics.regions.get_slice(space)));
    }

    debug!("subst_receiver_types_in_method_ty: method_generics={}",
           method_generics.repr(tcx));

    let method_fty = method.fty.subst(tcx, &combined_substs);

    debug!("subst_receiver_types_in_method_ty: method_ty={}",
           method.fty.repr(tcx));

    ty::Method::new(
        method.ident,
        method_generics,
        method_fty,
        method.explicit_self,
        method.vis,
        new_def_id,
        ImplContainer(impl_id),
        provided_source
    )
}

pub fn check_coherence(crate_context: &CrateCtxt) {
    CoherenceChecker {
        crate_context: crate_context,
        inference_context: new_infer_ctxt(crate_context.tcx),
        inherent_impls: RefCell::new(FnvHashMap::new()),
    }.check(crate_context.tcx.map.krate());
    orphan::check(crate_context.tcx);
    overlap::check(crate_context.tcx);
}
