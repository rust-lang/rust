// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
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
// The job of the coherence phase of typechecking is to ensure that each trait
// has at most one implementation for each type. Then we build a mapping from
// each trait in the system to its implementations.


use metadata::csearch::{each_impl, get_impl_trait, each_implementation_for_trait};
use metadata::csearch;
use middle::ty::get;
use middle::ty::{ImplContainer, lookup_item_type, subst};
use middle::ty::{substs, t, ty_bool, ty_char, ty_bot, ty_box, ty_enum, ty_err};
use middle::ty::{ty_str, ty_vec, ty_float, ty_infer, ty_int, ty_nil};
use middle::ty::{ty_param, ty_param_bounds_and_ty, ty_ptr};
use middle::ty::{ty_rptr, ty_self, ty_struct, ty_trait, ty_tup};
use middle::ty::{ty_uint, ty_uniq, ty_bare_fn, ty_closure};
use middle::ty::{ty_unboxed_vec, type_is_ty_var};
use middle::subst::Subst;
use middle::ty;
use middle::ty::{Impl, Method};
use middle::typeck::CrateCtxt;
use middle::typeck::infer::combine::Combine;
use middle::typeck::infer::InferCtxt;
use middle::typeck::infer::{new_infer_ctxt, resolve_ivar, resolve_type};
use middle::typeck::infer;
use util::ppaux::Repr;
use syntax::ast::{Crate, DefId, DefStruct, DefTy};
use syntax::ast::{Item, ItemEnum, ItemImpl, ItemMod, ItemStruct};
use syntax::ast::{LOCAL_CRATE, TraitRef, TyPath};
use syntax::ast;
use syntax::ast_map::NodeItem;
use syntax::ast_map;
use syntax::ast_util::{def_id_of_def, local_def};
use syntax::codemap::Span;
use syntax::opt_vec;
use syntax::parse::token;
use syntax::visit;

use std::cell::RefCell;
use collections::HashSet;
use std::rc::Rc;
use std::vec;

struct UniversalQuantificationResult {
    monotype: t,
    type_variables: ~[ty::t],
    type_param_defs: Rc<~[ty::TypeParameterDef]>
}

fn get_base_type(inference_context: &InferCtxt,
                 span: Span,
                 original_type: t)
                 -> Option<t> {
    let resolved_type;
    match resolve_type(inference_context,
                       original_type,
                       resolve_ivar) {
        Ok(resulting_type) if !type_is_ty_var(resulting_type) => {
            resolved_type = resulting_type;
        }
        _ => {
            inference_context.tcx.sess.span_fatal(span,
                                                  "the type of this value must be known in order \
                                                   to determine the base type");
        }
    }

    match get(resolved_type).sty {
        ty_enum(..) | ty_trait(..) | ty_struct(..) => {
            debug!("(getting base type) found base type");
            Some(resolved_type)
        }

        ty_nil | ty_bot | ty_bool | ty_char | ty_int(..) | ty_uint(..) | ty_float(..) |
        ty_str(..) | ty_vec(..) | ty_bare_fn(..) | ty_closure(..) | ty_tup(..) |
        ty_infer(..) | ty_param(..) | ty_self(..) |
        ty_unboxed_vec(..) | ty_err | ty_box(_) |
        ty_uniq(_) | ty_ptr(_) | ty_rptr(_, _) => {
            debug!("(getting base type) no base type; found {:?}",
                   get(original_type).sty);
            None
        }
    }
}

fn type_is_defined_in_local_crate(original_type: t) -> bool {
    /*!
     *
     * For coherence, when we have `impl Trait for Type`, we need to
     * guarantee that `Type` is "local" to the
     * crate.  For our purposes, this means that it must contain
     * some nominal type defined in this crate.
     */

    let mut found_nominal = false;
    ty::walk_ty(original_type, |t| {
        match get(t).sty {
            ty_enum(def_id, _) |
            ty_trait(def_id, _, _, _, _) |
            ty_struct(def_id, _) => {
                if def_id.krate == ast::LOCAL_CRATE {
                    found_nominal = true;
                }
            }

            _ => { }
        }
    });
    return found_nominal;
}

// Returns the def ID of the base type, if there is one.
fn get_base_type_def_id(inference_context: &InferCtxt,
                        span: Span,
                        original_type: t)
                        -> Option<DefId> {
    match get_base_type(inference_context, span, original_type) {
        None => {
            return None;
        }
        Some(base_type) => {
            match get(base_type).sty {
                ty_enum(def_id, _) |
                ty_struct(def_id, _) |
                ty_trait(def_id, _, _, _, _) => {
                    return Some(def_id);
                }
                _ => {
                    fail!("get_base_type() returned a type that wasn't an \
                           enum, struct, or trait");
                }
            }
        }
    }
}

struct CoherenceChecker {
    crate_context: @CrateCtxt,
    inference_context: InferCtxt,
}

struct CoherenceCheckVisitor<'a> {
    cc: &'a CoherenceChecker
}

impl<'a> visit::Visitor<()> for CoherenceCheckVisitor<'a> {
    fn visit_item(&mut self, item: &Item, _: ()) {

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

        visit::walk_item(self, item, ());
    }
}

struct PrivilegedScopeVisitor<'a> { cc: &'a CoherenceChecker }

impl<'a> visit::Visitor<()> for PrivilegedScopeVisitor<'a> {
    fn visit_item(&mut self, item: &Item, _: ()) {

        match item.node {
            ItemMod(ref module_) => {
                // Then visit the module items.
                visit::walk_mod(self, module_, ());
            }
            ItemImpl(_, None, ast_ty, _) => {
                if !self.cc.ast_type_is_defined_in_local_crate(ast_ty) {
                    // This is an error.
                    let session = self.cc.crate_context.tcx.sess;
                    session.span_err(item.span,
                                     "cannot associate methods with a type outside the \
                                     crate the type is defined in; define and implement \
                                     a trait or new type instead");
                }
            }
            ItemImpl(_, Some(ref trait_ref), _, _) => {
                // `for_ty` is `Type` in `impl Trait for Type`
                let for_ty =
                    ty::node_id_to_type(self.cc.crate_context.tcx,
                                        item.id);
                if !type_is_defined_in_local_crate(for_ty) {
                    // This implementation is not in scope of its base
                    // type. This still might be OK if the trait is
                    // defined in the same crate.

                    let trait_def_id =
                        self.cc.trait_ref_to_trait_def_id(trait_ref);

                    if trait_def_id.krate != LOCAL_CRATE {
                        let session = self.cc.crate_context.tcx.sess;
                        session.span_err(item.span,
                                "cannot provide an extension implementation \
                                where both trait and type are not defined in this crate");
                    }
                }

                visit::walk_item(self, item, ());
            }
            _ => {
                visit::walk_item(self, item, ());
            }
        }
    }
}

impl CoherenceChecker {
    fn new(crate_context: @CrateCtxt) -> CoherenceChecker {
        CoherenceChecker {
            crate_context: crate_context,
            inference_context: new_infer_ctxt(crate_context.tcx),
        }
    }

    fn check(&self, krate: &Crate) {
        // Check implementations and traits. This populates the tables
        // containing the inherent methods and extension methods. It also
        // builds up the trait inheritance table.
        let mut visitor = CoherenceCheckVisitor { cc: self };
        visit::walk_crate(&mut visitor, krate, ());

        // Check that there are no overlapping trait instances
        self.check_implementation_coherence();

        // Check whether traits with base types are in privileged scopes.
        self.check_privileged_scopes(krate);

        // Bring in external crates. It's fine for this to happen after the
        // coherence checks, because we ensure by construction that no errors
        // can happen at link time.
        self.add_external_crates();

        // Populate the table of destructors. It might seem a bit strange to
        // do this here, but it's actually the most convenient place, since
        // the coherence tables contain the trait -> type mappings.
        self.populate_destructor_table();
    }

    fn check_implementation(&self, item: &Item,
                            associated_traits: &[TraitRef]) {
        let tcx = self.crate_context.tcx;
        let self_type = ty::lookup_item_type(tcx, local_def(item.id));

        // If there are no traits, then this implementation must have a
        // base type.

        if associated_traits.len() == 0 {
            debug!("(checking implementation) no associated traits for item '{}'",
                   token::get_ident(item.ident));

            match get_base_type_def_id(&self.inference_context,
                                       item.span,
                                       self_type.ty) {
                None => {
                    let session = self.crate_context.tcx.sess;
                    session.span_err(item.span,
                                     "no base type found for inherent implementation; \
                                      implement a trait or new type instead");
                }
                Some(_) => {
                    // Nothing to do.
                }
            }
        }

        let implementation = self.create_impl_from_item(item);

        for associated_trait in associated_traits.iter() {
            let trait_ref = ty::node_id_to_trait_ref(
                self.crate_context.tcx, associated_trait.ref_id);
            debug!("(checking implementation) adding impl for trait '{}', item '{}'",
                   trait_ref.repr(self.crate_context.tcx),
                   token::get_ident(item.ident));

            self.add_trait_impl(trait_ref.def_id, implementation);
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
                    self.add_inherent_impl(base_type_def_id, implementation);
                }
            }
        }

        let mut impls = tcx.impls.borrow_mut();
        impls.get().insert(implementation.did, implementation);
    }

    // Creates default method IDs and performs type substitutions for an impl
    // and trait pair. Then, for each provided method in the trait, inserts a
    // `ProvidedMethodInfo` instance into the `provided_method_sources` map.
    fn instantiate_default_methods(&self, impl_id: ast::DefId,
                                   trait_ref: &ty::TraitRef,
                                   all_methods: &mut ~[@Method]) {
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
                @subst_receiver_types_in_method_ty(
                    tcx,
                    impl_id,
                    trait_ref,
                    new_did,
                    *trait_method,
                    Some(trait_method.def_id));

            debug!("new_method_ty={}", new_method_ty.repr(tcx));
            all_methods.push(new_method_ty);

            // construct the polytype for the method based on the method_ty
            let new_generics = ty::Generics {
                type_param_defs:
                    Rc::new(vec::append(
                        impl_poly_type.generics.type_param_defs().to_owned(),
                            new_method_ty.generics.type_param_defs())),
                region_param_defs:
                    impl_poly_type.generics.region_param_defs.clone()
            };
            let new_polytype = ty::ty_param_bounds_and_ty {
                generics: new_generics,
                ty: ty::mk_bare_fn(tcx, new_method_ty.fty.clone())
            };
            debug!("new_polytype={}", new_polytype.repr(tcx));

            {
                let mut tcache = tcx.tcache.borrow_mut();
                tcache.get().insert(new_did, new_polytype);
            }

            let mut methods = tcx.methods.borrow_mut();
            methods.get().insert(new_did, new_method_ty);

            // Pair the new synthesized ID up with the
            // ID of the method.
            let mut provided_method_sources =
                self.crate_context.tcx.provided_method_sources.borrow_mut();
            provided_method_sources.get().insert(new_did,
                                                 trait_method.def_id);
        }
    }

    fn add_inherent_impl(&self, base_def_id: DefId,
                         implementation: @Impl) {
        let tcx = self.crate_context.tcx;
        let implementation_list;
        let mut inherent_impls = tcx.inherent_impls.borrow_mut();
        match inherent_impls.get().find(&base_def_id) {
            None => {
                implementation_list = @RefCell::new(~[]);
                inherent_impls.get().insert(base_def_id, implementation_list);
            }
            Some(&existing_implementation_list) => {
                implementation_list = existing_implementation_list;
            }
        }

        let mut implementation_list = implementation_list.borrow_mut();
        implementation_list.get().push(implementation);
    }

    fn add_trait_impl(&self, base_def_id: DefId,
                      implementation: @Impl) {
        let tcx = self.crate_context.tcx;
        let implementation_list;
        let mut trait_impls = tcx.trait_impls.borrow_mut();
        match trait_impls.get().find(&base_def_id) {
            None => {
                implementation_list = @RefCell::new(~[]);
                trait_impls.get().insert(base_def_id, implementation_list);
            }
            Some(&existing_implementation_list) => {
                implementation_list = existing_implementation_list;
            }
        }

        let mut implementation_list = implementation_list.borrow_mut();
        implementation_list.get().push(implementation);
    }

    fn check_implementation_coherence(&self) {
        let trait_impls = self.crate_context.tcx.trait_impls.borrow();
        for &trait_id in trait_impls.get().keys() {
            self.check_implementation_coherence_of(trait_id);
        }
    }

    fn check_implementation_coherence_of(&self, trait_def_id: DefId) {
        // Unify pairs of polytypes.
        self.iter_impls_of_trait_local(trait_def_id, |a| {
            let implementation_a = a;
            let polytype_a =
                self.get_self_type_for_implementation(implementation_a);

            // "We have an impl of trait <trait_def_id> for type <polytype_a>,
            // and that impl is <implementation_a>"
            self.iter_impls_of_trait(trait_def_id, |b| {
                let implementation_b = b;

                // An impl is coherent with itself
                if a.did != b.did {
                    let polytype_b = self.get_self_type_for_implementation(
                            implementation_b);

                    if self.polytypes_unify(polytype_a.clone(), polytype_b) {
                        let session = self.crate_context.tcx.sess;
                        session.span_err(
                            self.span_of_impl(implementation_a),
                            format!("conflicting implementations for trait `{}`",
                                 ty::item_path_str(self.crate_context.tcx,
                                                   trait_def_id)));
                        if implementation_b.did.krate == LOCAL_CRATE {
                            session.span_note(self.span_of_impl(implementation_b),
                                              "note conflicting implementation here");
                        } else {
                            let crate_store = self.crate_context.tcx.sess.cstore;
                            let cdata = crate_store.get_crate_data(implementation_b.did.krate);
                            session.note(
                                "conflicting implementation in crate `" + cdata.name + "`");
                        }
                    }
                }
            })
        })
    }

    fn iter_impls_of_trait(&self, trait_def_id: DefId, f: |@Impl|) {
        self.iter_impls_of_trait_local(trait_def_id, |x| f(x));

        if trait_def_id.krate == LOCAL_CRATE {
            return;
        }

        let crate_store = self.crate_context.tcx.sess.cstore;
        csearch::each_implementation_for_trait(crate_store, trait_def_id, |impl_def_id| {
            let implementation = @csearch::get_impl(self.crate_context.tcx, impl_def_id);
            let _ = lookup_item_type(self.crate_context.tcx, implementation.did);
            f(implementation);
        });
    }

    fn iter_impls_of_trait_local(&self, trait_def_id: DefId, f: |@Impl|) {
        let trait_impls = self.crate_context.tcx.trait_impls.borrow();
        match trait_impls.get().find(&trait_def_id) {
            Some(impls) => {
                let impls = impls.borrow();
                for &im in impls.get().iter() {
                    f(im);
                }
            }
            None => { /* no impls? */ }
        }
    }

    fn polytypes_unify(&self,
                       polytype_a: ty_param_bounds_and_ty,
                       polytype_b: ty_param_bounds_and_ty)
                       -> bool {
        let universally_quantified_a =
            self.universally_quantify_polytype(polytype_a);
        let universally_quantified_b =
            self.universally_quantify_polytype(polytype_b);

        return self.can_unify_universally_quantified(
            &universally_quantified_a, &universally_quantified_b) ||
            self.can_unify_universally_quantified(
            &universally_quantified_b, &universally_quantified_a);
    }

    // Converts a polytype to a monotype by replacing all parameters with
    // type variables. Returns the monotype and the type variables created.
    fn universally_quantify_polytype(&self, polytype: ty_param_bounds_and_ty)
                                     -> UniversalQuantificationResult {
        let region_parameter_count = polytype.generics.region_param_defs().len();
        let region_parameters =
            self.inference_context.next_region_vars(
                infer::BoundRegionInCoherence,
                region_parameter_count);

        let bounds_count = polytype.generics.type_param_defs().len();
        let type_parameters = self.inference_context.next_ty_vars(bounds_count);

        let substitutions = substs {
            regions: ty::NonerasedRegions(opt_vec::from(region_parameters)),
            self_ty: None,
            tps: type_parameters
        };
        let monotype = subst(self.crate_context.tcx,
                             &substitutions,
                             polytype.ty);

        UniversalQuantificationResult {
            monotype: monotype,
            type_variables: substitutions.tps,
            type_param_defs: polytype.generics.type_param_defs.clone()
        }
    }

    fn can_unify_universally_quantified<'a>(&self,
                                            a: &'a UniversalQuantificationResult,
                                            b: &'a UniversalQuantificationResult)
                                            -> bool {
        infer::can_mk_subty(&self.inference_context,
                            a.monotype,
                            b.monotype).is_ok()
    }

    fn get_self_type_for_implementation(&self, implementation: @Impl)
                                        -> ty_param_bounds_and_ty {
        let tcache = self.crate_context.tcx.tcache.borrow();
        return tcache.get().get_copy(&implementation.did);
    }

    // Privileged scope checking
    fn check_privileged_scopes(&self, krate: &Crate) {
        let mut visitor = PrivilegedScopeVisitor{ cc: self };
        visit::walk_crate(&mut visitor, krate, ());
    }

    fn trait_ref_to_trait_def_id(&self, trait_ref: &TraitRef) -> DefId {
        let def_map = self.crate_context.tcx.def_map;
        let def_map = def_map.borrow();
        let trait_def = def_map.get().get_copy(&trait_ref.ref_id);
        let trait_id = def_id_of_def(trait_def);
        return trait_id;
    }

    /// For coherence, when we have `impl Type`, we need to guarantee that
    /// `Type` is "local" to the crate. For our purposes, this means that it
    /// must precisely name some nominal type defined in this crate.
    fn ast_type_is_defined_in_local_crate(&self, original_type: &ast::Ty) -> bool {
        match original_type.node {
            TyPath(_, _, path_id) => {
                let def_map = self.crate_context.tcx.def_map.borrow();
                match def_map.get().get_copy(&path_id) {
                    DefTy(def_id) | DefStruct(def_id) => {
                        if def_id.krate != LOCAL_CRATE {
                            return false;
                        }

                        // Make sure that this type precisely names a nominal
                        // type.
                        match self.crate_context.tcx.map.find(def_id.node) {
                            None => {
                                self.crate_context.tcx.sess.span_bug(
                                    original_type.span,
                                    "resolve didn't resolve this type?!");
                            }
                            Some(NodeItem(item)) => {
                                match item.node {
                                    ItemStruct(..) | ItemEnum(..) => true,
                                    _ => false,
                                }
                            }
                            Some(_) => false,
                        }
                    }
                    _ => false
                }
            }
            _ => false
        }
    }

    // Converts an implementation in the AST to an Impl structure.
    fn create_impl_from_item(&self, item: &Item) -> @Impl {
        let tcx = self.crate_context.tcx;
        match item.node {
            ItemImpl(_, ref trait_refs, _, ref ast_methods) => {
                let mut methods = ~[];
                for ast_method in ast_methods.iter() {
                    methods.push(ty::method(tcx, local_def(ast_method.id)));
                }

                for trait_ref in trait_refs.iter() {
                    let ty_trait_ref = ty::node_id_to_trait_ref(
                        self.crate_context.tcx,
                        trait_ref.ref_id);

                    self.instantiate_default_methods(local_def(item.id),
                                                     ty_trait_ref,
                                                     &mut methods);
                }

                return @Impl {
                    did: local_def(item.id),
                    ident: item.ident,
                    methods: methods
                };
            }
            _ => {
                self.crate_context.tcx.sess.span_bug(item.span,
                                                     "can't convert a non-impl to an impl");
            }
        }
    }

    fn span_of_impl(&self, implementation: @Impl) -> Span {
        assert_eq!(implementation.did.krate, LOCAL_CRATE);
        self.crate_context.tcx.map.span(implementation.did.node)
    }

    // External crate handling

    fn add_external_impl(&self,
                         impls_seen: &mut HashSet<DefId>,
                         impl_def_id: DefId) {
        let tcx = self.crate_context.tcx;
        let implementation = @csearch::get_impl(tcx, impl_def_id);

        // Make sure we don't visit the same implementation multiple times.
        if !impls_seen.insert(implementation.did) {
            // Skip this one.
            return
        }
        // Good. Continue.

        let _ = lookup_item_type(tcx, implementation.did);
        let associated_traits = get_impl_trait(tcx, implementation.did);

        // Do a sanity check.
        assert!(associated_traits.is_some());

        // Record all the trait methods.
        for trait_ref in associated_traits.iter() {
              self.add_trait_impl(trait_ref.def_id, implementation);
        }

        // For any methods that use a default implementation, add them to
        // the map. This is a bit unfortunate.
        for method in implementation.methods.iter() {
            for source in method.provided_source.iter() {
                let mut provided_method_sources = tcx.provided_method_sources
                                                     .borrow_mut();
                provided_method_sources.get().insert(method.def_id, *source);
            }
        }

        let mut impls = tcx.impls.borrow_mut();
        impls.get().insert(implementation.did, implementation);
    }

    // Adds implementations and traits from external crates to the coherence
    // info.
    fn add_external_crates(&self) {
        let mut impls_seen = HashSet::new();

        let crate_store = self.crate_context.tcx.sess.cstore;
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

        let trait_impls = tcx.trait_impls.borrow();
        let impls_opt = trait_impls.get().find(&drop_trait);
        let impls;
        match impls_opt {
            None => return, // No types with (new-style) dtors present.
            Some(found_impls) => impls = found_impls
        }

        let impls = impls.borrow();
        for impl_info in impls.get().iter() {
            if impl_info.methods.len() < 1 {
                // We'll error out later. For now, just don't ICE.
                continue;
            }
            let method_def_id = impl_info.methods[0].def_id;

            let self_type = self.get_self_type_for_implementation(*impl_info);
            match ty::get(self_type.ty).sty {
                ty::ty_enum(type_def_id, _) |
                ty::ty_struct(type_def_id, _) => {
                    let mut destructor_for_type = tcx.destructor_for_type
                                                     .borrow_mut();
                    destructor_for_type.get().insert(type_def_id,
                                                     method_def_id);
                    let mut destructors = tcx.destructors.borrow_mut();
                    destructors.get().insert(method_def_id);
                }
                _ => {
                    // Destructors only work on nominal types.
                    if impl_info.did.krate == ast::LOCAL_CRATE {
                        {
                            match tcx.map.find(impl_info.did.node) {
                                Some(ast_map::NodeItem(item)) => {
                                    tcx.sess.span_err((*item).span,
                                                      "the Drop trait may \
                                                       only be implemented \
                                                       on structures");
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

pub fn make_substs_for_receiver_types(tcx: ty::ctxt,
                                      impl_id: ast::DefId,
                                      trait_ref: &ty::TraitRef,
                                      method: &ty::Method)
                                      -> ty::substs {
    /*!
     * Substitutes the values for the receiver's type parameters
     * that are found in method, leaving the method's type parameters
     * intact.  This is in fact a mildly complex operation,
     * largely because of the hokey way that we concatenate the
     * receiver and method generics.
     */

    // determine how many type parameters were declared on the impl
    let num_impl_type_parameters = {
        let impl_polytype = ty::lookup_item_type(tcx, impl_id);
        impl_polytype.generics.type_param_defs().len()
    };

    // determine how many type parameters appear on the trait
    let num_trait_type_parameters = trait_ref.substs.tps.len();

    // the current method type has the type parameters from the trait + method
    let num_method_type_parameters =
        num_trait_type_parameters + method.generics.type_param_defs().len();

    // the new method type will have the type parameters from the impl + method
    let combined_tps = vec::from_fn(num_method_type_parameters, |i| {
        if i < num_trait_type_parameters {
            // replace type parameters that come from trait with new value
            trait_ref.substs.tps[i]
        } else {
            // replace type parameters that belong to method with another
            // type parameter, this time with the index adjusted
            let method_index = i - num_trait_type_parameters;
            let type_param_def = &method.generics.type_param_defs()[method_index];
            let new_index = num_impl_type_parameters + method_index;
            ty::mk_param(tcx, new_index, type_param_def.def_id)
        }
    });

    return ty::substs {
        regions: trait_ref.substs.regions.clone(),
        self_ty: trait_ref.substs.self_ty,
        tps: combined_tps
    };
}

fn subst_receiver_types_in_method_ty(tcx: ty::ctxt,
                                     impl_id: ast::DefId,
                                     trait_ref: &ty::TraitRef,
                                     new_def_id: ast::DefId,
                                     method: &ty::Method,
                                     provided_source: Option<ast::DefId>)
                                     -> ty::Method {

    let combined_substs = make_substs_for_receiver_types(
        tcx, impl_id, trait_ref, method);

    ty::Method::new(
        method.ident,

        // method types *can* appear in the generic bounds
        method.generics.subst(tcx, &combined_substs),

        // method types *can* appear in the fty
        method.fty.subst(tcx, &combined_substs),

        method.explicit_self,
        method.vis,
        new_def_id,
        ImplContainer(impl_id),
        provided_source
    )
}

pub fn check_coherence(crate_context: @CrateCtxt, krate: &Crate) {
    CoherenceChecker::new(crate_context).check(krate);
}
