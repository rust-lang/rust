// Coherence phase
//
// The job of the coherence phase of typechecking is to ensure that each trait
// has at most one implementation for each type. Then we build a mapping from
// each trait in the system to its implementations.

import metadata::csearch::{each_path, get_impl_traits, get_impls_for_mod};
import metadata::cstore::{cstore, iter_crate_data};
import metadata::decoder::{dl_def, dl_field, dl_impl};
import middle::resolve3::Impl;
import middle::ty::{get, lookup_item_type, subst, t, ty_box};
import middle::ty::{ty_uniq, ty_ptr, ty_rptr, ty_enum};
import middle::ty::{ty_class, ty_nil, ty_bot, ty_bool, ty_int, ty_uint};
import middle::ty::{ty_float, ty_estr, ty_evec, ty_rec};
import middle::ty::{ty_fn, ty_trait, ty_tup, ty_var, ty_var_integral};
import middle::ty::{ty_param, ty_self, ty_type, ty_opaque_box};
import middle::ty::{ty_opaque_closure_ptr, ty_unboxed_vec, type_is_var};
import middle::typeck::infer::{infer_ctxt, mk_subty};
import middle::typeck::infer::{new_infer_ctxt, resolve_ivar, resolve_type};
import syntax::ast::{crate, def_id, def_mod, item, item_class, item_const};
import syntax::ast::{item_enum, item_fn, item_foreign_mod, item_impl};
import syntax::ast::{item_mac, item_mod, item_trait, item_ty, local_crate};
import syntax::ast::{method, node_id, region_param, rp_none, rp_self};
import syntax::ast::{trait_ref};
import syntax::ast_map::node_item;
import syntax::ast_util::{def_id_of_def, dummy_sp, new_def_hash};
import syntax::codemap::span;
import syntax::visit::{default_simple_visitor, default_visitor};
import syntax::visit::{mk_simple_visitor, mk_vt, visit_crate, visit_item};
import syntax::visit::{visit_mod};
import util::ppaux::ty_to_str;

import dvec::{dvec, extensions};
import result::{extensions, ok};
import std::map::{hashmap, int_hash};
import uint::range;
import vec::{len, push};

fn get_base_type(inference_context: infer_ctxt, span: span, original_type: t)
              -> option<t> {

    let resolved_type;
    alt resolve_type(inference_context,
                     original_type,
                     resolve_ivar) {
        ok(resulting_type) if !type_is_var(resulting_type) {
            resolved_type = resulting_type;
        }
        _ {
            inference_context.tcx.sess.span_fatal(span,
                                                  ~"the type of this value \
                                                    must be known in order \
                                                    to determine the base \
                                                    type");
        }
    }

    alt get(resolved_type).struct {
        ty_box(base_mutability_and_type) |
        ty_uniq(base_mutability_and_type) |
        ty_ptr(base_mutability_and_type) |
        ty_rptr(_, base_mutability_and_type) {
            #debug("(getting base type) recurring");
            get_base_type(inference_context, span,
                          base_mutability_and_type.ty)
        }

        ty_enum(*) | ty_trait(*) | ty_class(*) {
            #debug("(getting base type) found base type");
            some(resolved_type)
        }

        ty_nil | ty_bot | ty_bool | ty_int(*) | ty_uint(*) | ty_float(*) |
        ty_estr(*) | ty_evec(*) | ty_rec(*) |
        ty_fn(*) | ty_tup(*) | ty_var(*) | ty_var_integral(*) |
        ty_param(*) | ty_self | ty_type | ty_opaque_box |
        ty_opaque_closure_ptr(*) | ty_unboxed_vec(*) {
            #debug("(getting base type) no base type; found %?",
                   get(original_type).struct);
            none
        }
    }
}

// Returns the def ID of the base type, if there is one.
fn get_base_type_def_id(inference_context: infer_ctxt,
                        span: span,
                        original_type: t)
                     -> option<def_id> {

    alt get_base_type(inference_context, span, original_type) {
        none {
            ret none;
        }
        some(base_type) {
            alt get(base_type).struct {
                ty_enum(def_id, _) |
                ty_class(def_id, _) |
                ty_trait(def_id, _) {
                    ret some(def_id);
                }
                _ {
                    fail ~"get_base_type() returned a type that wasn't an \
                           enum, class, or trait";
                }
            }
        }
    }
}

class CoherenceInfo {
    // Contains implementations of methods that are inherent to a type.
    // Methods in these implementations don't need to be exported.

    let inherent_methods: hashmap<def_id,@dvec<@Impl>>;

    // Contains implementations of methods associated with a trait. For these,
    // the associated trait must be imported at the call site.
    let extension_methods: hashmap<def_id,@dvec<@Impl>>;

    new() {
        self.inherent_methods = new_def_hash();
        self.extension_methods = new_def_hash();
    }
}

class CoherenceChecker {
    let crate_context: @crate_ctxt;
    let inference_context: infer_ctxt;

    // A mapping from implementations to the corresponding base type
    // definition ID.

    let base_type_def_ids: hashmap<def_id,def_id>;

    // A set of implementations in privileged scopes; i.e. those
    // implementations that are defined in the same scope as their base types.

    let privileged_implementations: hashmap<node_id,()>;

    // The set of types that we are currently in the privileged scope of. This
    // is used while we traverse the AST while checking privileged scopes.

    let privileged_types: hashmap<def_id,()>;

    new(crate_context: @crate_ctxt) {
        self.crate_context = crate_context;
        self.inference_context = new_infer_ctxt(crate_context.tcx);

        self.base_type_def_ids = new_def_hash();
        self.privileged_implementations = int_hash();
        self.privileged_types = new_def_hash();
    }

    fn check_coherence(crate: @crate) {
        // Check implementations. This populates the tables containing the
        // inherent methods and extension methods.

        visit_crate(*crate, (), mk_simple_visitor(@{
            visit_item: |item| {
                #debug("(checking coherence) item '%s'", *item.ident);

                alt item.node {
                    item_impl(_, associated_traits, self_type, _) {
                        self.check_implementation(item, associated_traits);
                    }
                    _ {
                        // Nothing to do.
                    }
                };
            }
            with *default_simple_visitor()
        }));

        // Check trait coherence.
        for self.crate_context.coherence_info.extension_methods.each
                |def_id, items| {

            self.check_implementation_coherence(def_id, items);
        }

        // Check whether traits with base types are in privileged scopes.
        self.check_privileged_scopes(crate);

        // Bring in external crates. It's fine for this to happen after the
        // coherence checks, because we ensure by construction that no errors
        // can happen at link time.

        self.add_external_crates();
    }

    fn check_implementation(item: @item, associated_traits: ~[@trait_ref]) {
        let self_type = self.crate_context.tcx.tcache.get(local_def(item.id));

        // If there are no traits, then this implementation must have a
        // base type.

        if associated_traits.len() == 0 {
            #debug("(checking implementation) no associated traits for item \
                    '%s'",
                   *item.ident);

            alt get_base_type_def_id(self.inference_context,
                                     item.span,
                                     self_type.ty) {
                none {
                    let session = self.crate_context.tcx.sess;
                    session.span_err(item.span,
                                     ~"no base type found for inherent \
                                       implementation; implement a \
                                       trait instead");
                }
                some(_) {
                    // Nothing to do.
                }
            }
        }

        for associated_traits.each |associated_trait| {
            let def = self.crate_context.tcx.def_map.get
                (associated_trait.ref_id);
            #debug("(checking implementation) adding impl for trait \
                    '%s', item '%s'",
                   ast_map::node_id_to_str(self.crate_context.tcx.items,
                                           associated_trait.ref_id),
                   *item.ident);

            let implementation = self.create_impl_from_item(item);
            self.add_trait_method(def_id_of_def(def), implementation);
        }

        // Add the implementation to the mapping from implementation to base
        // type def ID, if there is a base type for this implementation.

        alt get_base_type_def_id(self.inference_context,
                                 item.span,
                                 self_type.ty) {
            none {
                // Nothing to do.
            }
            some(base_type_def_id) {
                let implementation = self.create_impl_from_item(item);
                self.add_inherent_method(base_type_def_id, implementation);

                self.base_type_def_ids.insert(local_def(item.id),
                                              base_type_def_id);
            }
        }
    }

    fn add_inherent_method(base_def_id: def_id, implementation: @Impl) {
        let implementation_list;
        alt self.crate_context.coherence_info.inherent_methods
            .find(base_def_id) {

            none {
                implementation_list = @dvec();
                self.crate_context.coherence_info.inherent_methods
                    .insert(base_def_id, implementation_list);
            }
            some(existing_implementation_list) {
                implementation_list = existing_implementation_list;
            }
        }

        implementation_list.push(implementation);
    }

    fn add_trait_method(trait_id: def_id, implementation: @Impl) {
        let implementation_list;
        alt self.crate_context.coherence_info.extension_methods
                .find(trait_id) {

            none {
                implementation_list = @dvec();
                self.crate_context.coherence_info.extension_methods
                    .insert(trait_id, implementation_list);
            }
            some(existing_implementation_list) {
                implementation_list = existing_implementation_list;
            }
        }

        implementation_list.push(implementation);
    }

    fn check_implementation_coherence(_trait_def_id: def_id,
                                      implementations: @dvec<@Impl>) {

        // Unify pairs of polytypes.
        for range(0, implementations.len()) |i| {
            let implementation_a = implementations.get_elt(i);
            let polytype_a =
                self.get_self_type_for_implementation(implementation_a);
            for range(i + 1, implementations.len()) |j| {
                let implementation_b = implementations.get_elt(j);
                let polytype_b =
                    self.get_self_type_for_implementation(implementation_b);

                if self.polytypes_unify(polytype_a, polytype_b) {
                    let session = self.crate_context.tcx.sess;
                    session.span_err(self.span_of_impl(implementation_b),
                                     ~"conflicting implementations for a \
                                       trait");
                    session.span_note(self.span_of_impl(implementation_a),
                                      ~"note conflicting implementation \
                                        here");
                }
            }
        }
    }

    fn polytypes_unify(polytype_a: ty_param_bounds_and_ty,
                       polytype_b: ty_param_bounds_and_ty)
                    -> bool {

        let monotype_a = self.universally_quantify_polytype(polytype_a);
        let monotype_b = self.universally_quantify_polytype(polytype_b);
        ret mk_subty(self.inference_context, monotype_a, monotype_b).is_ok()
         || mk_subty(self.inference_context, monotype_b, monotype_a).is_ok();
    }

    // Converts a polytype to a monotype by replacing all parameters with
    // type variables.

    fn universally_quantify_polytype(polytype: ty_param_bounds_and_ty) -> t {
        let self_region =
            if !polytype.rp {none}
            else {some(self.inference_context.next_region_var_nb())};

        let bounds_count = polytype.bounds.len();
        let type_parameters =
            self.inference_context.next_ty_vars(bounds_count);

        let substitutions = {
            self_r: self_region,
            self_ty: none,
            tps: type_parameters
        };

        ret subst(self.crate_context.tcx, substitutions, polytype.ty);
    }

    fn get_self_type_for_implementation(implementation: @Impl)
                                     -> ty_param_bounds_and_ty {

        ret self.crate_context.tcx.tcache.get(implementation.did);
    }

    // Privileged scope checking

    fn check_privileged_scopes(crate: @crate) {
        // Gather up all privileged types.
        let privileged_types =
            self.gather_privileged_types(crate.node.module.items);
        for privileged_types.each |privileged_type| {
            self.privileged_types.insert(privileged_type, ());
        }

        visit_crate(*crate, (), mk_vt(@{
            visit_item: |item, _context, visitor| {
                alt item.node {
                    item_mod(module) {
                        // First, gather up all privileged types.
                        let privileged_types =
                            self.gather_privileged_types(module.items);
                        for privileged_types.each |privileged_type| {
                            #debug("(checking privileged scopes) entering \
                                    privileged scope of %d:%d",
                                   privileged_type.crate,
                                   privileged_type.node);

                            self.privileged_types.insert(privileged_type, ());
                        }

                        // Then visit the module items.
                        visit_mod(module, item.span, item.id, (), visitor);

                        // Finally, remove privileged types from the map.
                        for privileged_types.each |privileged_type| {
                            self.privileged_types.remove(privileged_type);
                        }
                    }
                    item_impl(_, associated_traits, _, _) {
                        alt self.base_type_def_ids.find(local_def(item.id)) {
                            none {
                                // Nothing to do.
                            }
                            some(base_type_def_id) {
                                // Check to see whether the implementation is
                                // in the scope of its base type.

                                let privileged_types = &self.privileged_types;
                                if privileged_types.
                                        contains_key(base_type_def_id) {

                                    // Record that this implementation is OK.
                                    self.privileged_implementations.insert
                                        (item.id, ());
                                } else {
                                    // This implementation is not in scope of
                                    // its base type. This still might be OK
                                    // if the traits are defined in the same
                                    // crate.

                                    if associated_traits.len() == 0 {
                                        // There is no trait to implement, so
                                        // this is an error.

                                        let session =
                                            self.crate_context.tcx.sess;
                                        session.span_err(item.span,
                                                         ~"cannot implement \
                                                          inherent methods \
                                                          for a type outside \
                                                          the scope the type \
                                                          was defined in; \
                                                          define and \
                                                          implement a trait \
                                                          instead");
                                    }

                                    for associated_traits.each |trait_ref| {
                                        // This is OK if and only if the
                                        // trait was defined in this
                                        // crate.

                                        let def_map = self.crate_context.tcx
                                            .def_map;
                                        let trait_def = def_map.get
                                            (trait_ref.ref_id);
                                        let trait_id =
                                            def_id_of_def(trait_def);
                                        if trait_id.crate != local_crate {
                                            let session = self.crate_context
                                                .tcx.sess;
                                            session.span_err(item.span,
                                                             ~"cannot \
                                                               provide an \
                                                               extension \
                                                               implementa\
                                                                  tion \
                                                               for a trait \
                                                               not defined \
                                                               in this \
                                                               crate");
                                        }
                                    }
                                }
                            }
                        }

                        visit_item(item, (), visitor);
                    }
                    _ {
                        visit_item(item, (), visitor);
                    }
                }
            }
            with *default_visitor()
        }));
    }

    fn gather_privileged_types(items: ~[@item]) -> @dvec<def_id> {
        let results = @dvec();
        for items.each |item| {
            alt item.node {
                item_class(*) | item_enum(*) | item_trait(*) {
                    results.push(local_def(item.id));
                }

                item_const(*) | item_fn(*) | item_mod(*) |
                item_foreign_mod(*) | item_ty(*) | item_impl(*) |
                item_mac(*) {
                    // Nothing to do.
                }
            }
        }

        ret results;
    }

    // Converts an implementation in the AST to an Impl structure.
    fn create_impl_from_item(item: @item) -> @Impl {
        alt item.node {
            item_impl(ty_params, _, _, ast_methods) {
                let mut methods = ~[];
                for ast_methods.each |ast_method| {
                    push(methods, @{
                        did: local_def(ast_method.id),
                        n_tps: ast_method.tps.len(),
                        ident: ast_method.ident
                    });
                }

                ret @{
                    did: local_def(item.id),
                    ident: item.ident,
                    methods: methods
                };
            }
            _ {
                self.crate_context.tcx.sess.span_bug(item.span,
                                                     ~"can't convert a \
                                                       non-impl to an impl");
            }
        }
    }

    fn span_of_impl(implementation: @Impl) -> span {
        assert implementation.did.crate == local_crate;
        alt self.crate_context.tcx.items.find(implementation.did.node) {
            some(node_item(item, _)) {
                ret item.span;
            }
            _ {
                self.crate_context.tcx.sess.bug(~"span_of_impl() called on \
                                                  something that wasn't an \
                                                  impl!");
            }
        }
    }

    // External crate handling

    fn add_impls_for_module(impls_seen: hashmap<def_id,()>,
                            crate_store: cstore,
                            module_def_id: def_id) {

        let implementations = get_impls_for_mod(crate_store,
                                                module_def_id,
                                                none);
        for (*implementations).each |implementation| {
            // Make sure we don't visit the same implementation
            // multiple times.
            alt impls_seen.find(implementation.did) {
                none {
                    // Good. Continue.
                    impls_seen.insert(implementation.did, ());
                }
                some(_) {
                    // Skip this one.
                    again;
                }
            }

            let self_type = lookup_item_type(self.crate_context.tcx,
                                             implementation.did);
            let associated_traits = get_impl_traits(self.crate_context.tcx,
                                                    implementation.did);

            // Do a sanity check to make sure that inherent methods have base
            // types.

            if associated_traits.len() == 0 {
                alt get_base_type_def_id(self.inference_context,
                                         dummy_sp(),
                                         self_type.ty) {
                    none {
                        let session = self.crate_context.tcx.sess;
                        session.bug(#fmt("no base type for external impl \
                                          with no trait: %s (type %s)!",
                                         *implementation.ident,
                                         ty_to_str(self.crate_context.tcx,
                                                   self_type.ty)));
                    }
                    some(_) {
                        // Nothing to do.
                    }
                }
            }

            // Record all the trait methods.
            for associated_traits.each |trait_type| {
                alt get(trait_type).struct {
                    ty_trait(trait_id, _) {
                        self.add_trait_method(trait_id, implementation);
                    }
                    _ {
                        self.crate_context.tcx.sess.bug(~"trait type \
                                                          returned is not a \
                                                          trait");
                    }
                }
            }

            // Add the implementation to the mapping from
            // implementation to base type def ID, if there is a base
            // type for this implementation.

            alt get_base_type_def_id(self.inference_context,
                                     dummy_sp(),
                                     self_type.ty) {
                none {
                    // Nothing to do.
                }
                some(base_type_def_id) {
                    self.add_inherent_method(base_type_def_id,
                                             implementation);

                    self.base_type_def_ids.insert(implementation.did,
                                                  base_type_def_id);
                }
            }
        }
    }

    fn add_external_crates() {
        let impls_seen = new_def_hash();

        let crate_store = self.crate_context.tcx.sess.cstore;
        do iter_crate_data(crate_store) |crate_number, _crate_metadata| {
            self.add_impls_for_module(impls_seen,
                                      crate_store,
                                      { crate: crate_number, node: 0 });

            for each_path(crate_store, crate_number) |path_entry| {
                let module_def_id;
                alt path_entry.def_like {
                    dl_def(def_mod(def_id)) {
                        module_def_id = def_id;
                    }
                    dl_def(_) | dl_impl(_) | dl_field {
                        // Skip this.
                        again;
                    }
                }

                self.add_impls_for_module(impls_seen,
                                          crate_store,
                                          module_def_id);
            }
        }
    }
}

fn check_coherence(crate_context: @crate_ctxt, crate: @crate) {
    CoherenceChecker(crate_context).check_coherence(crate);
}

