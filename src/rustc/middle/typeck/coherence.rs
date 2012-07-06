// Coherence phase
//
// The job of the coherence phase of typechecking is to ensure that each trait
// has at most one implementation for each type. Then we build a mapping from
// each trait in the system to its implementations.

import middle::ty::{get, t, ty_box, ty_uniq, ty_ptr, ty_rptr, ty_enum};
import middle::ty::{ty_class, ty_nil, ty_bot, ty_bool, ty_int, ty_uint};
import middle::ty::{ty_float, ty_estr, ty_evec, ty_rec};
import middle::ty::{ty_fn, ty_trait, ty_tup, ty_var, ty_var_integral};
import middle::ty::{ty_param, ty_self, ty_type, ty_opaque_box};
import middle::ty::{ty_opaque_closure_ptr, ty_unboxed_vec, new_ty_hash};
import middle::ty::{subst};
import middle::typeck::infer::{infer_ctxt, mk_subty, new_infer_ctxt};
import syntax::ast::{crate, def_id, item, item_class, item_const, item_enum};
import syntax::ast::{item_fn, item_foreign_mod, item_impl, item_mac};
import syntax::ast::{item_mod, item_trait, item_ty, local_crate, method};
import syntax::ast::{node_id, trait_ref};
import syntax::ast_util::{def_id_of_def, new_def_hash};
import syntax::visit::{default_simple_visitor, default_visitor};
import syntax::visit::{mk_simple_visitor, mk_vt, visit_crate, visit_item};
import syntax::visit::{visit_mod};
import util::ppaux::ty_to_str;

import dvec::{dvec, extensions};
import result::{extensions};
import std::map::{hashmap, int_hash};
import uint::range;

class CoherenceInfo {
    // Contains implementations of methods that are inherent to a type.
    // Methods in these implementations don't need to be exported.
    let inherent_methods: hashmap<t,@dvec<@item>>;

    // Contains implementations of methods associated with a trait. For these,
    // the associated trait must be imported at the call site.
    let extension_methods: hashmap<def_id,@dvec<@item>>;

    new() {
        self.inherent_methods = new_ty_hash();
        self.extension_methods = new_def_hash();
    }
}

class CoherenceChecker {
    let crate_context: @crate_ctxt;
    let inference_context: infer_ctxt;
    let info: @CoherenceInfo;

    // A mapping from implementations to the corresponding base type
    // definition ID.
    let base_type_def_ids: hashmap<node_id,def_id>;

    // A set of implementations in privileged scopes; i.e. those
    // implementations that are defined in the same scope as their base types.
    let privileged_implementations: hashmap<node_id,()>;

    // The set of types that we are currently in the privileged scope of. This
    // is used while we traverse the AST while checking privileged scopes.
    let privileged_types: hashmap<def_id,()>;

    new(crate_context: @crate_ctxt) {
        self.crate_context = crate_context;
        self.inference_context = new_infer_ctxt(crate_context.tcx);
        self.info = @CoherenceInfo();

        self.base_type_def_ids = int_hash();
        self.privileged_implementations = int_hash();
        self.privileged_types = new_def_hash();
    }

    fn check_coherence(crate: @crate) {
        // Check implementations. This populates the tables containing the
        // inherent methods and extension methods.

        visit_crate(*crate, (), mk_simple_visitor(@{
            visit_item: |item| {
                alt item.node {
                    item_impl(_, associated_trait, self_type, _) {
                        self.check_implementation(item, associated_trait);
                    }
                    _ {
                        // Nothing to do.
                    }
                };
            }
            with *default_simple_visitor()
        }));

        // Check trait coherence.
        for self.info.extension_methods.each |def_id, items| {
            self.check_implementation_coherence(def_id, items);
        }

        // Check whether traits with base types are in privileged scopes.
        self.check_privileged_scopes(crate);
    }

    fn check_implementation(item: @item,
                            optional_associated_trait: option<@trait_ref>) {

        let self_type = self.crate_context.tcx.tcache.get(local_def(item.id));
        alt optional_associated_trait {
            none {
                alt self.get_base_type(self_type.ty) {
                    none {
                        let session = self.crate_context.tcx.sess;
                        session.span_warn(item.span,
                                          ~"no base type found for inherent \
                                           implementation; implement a trait \
                                           instead");
                    }
                    some(base_type) {
                        let implementation_list;
                        alt self.info.inherent_methods.find(base_type) {
                            none {
                                implementation_list = @dvec();
                            }
                            some(existing_implementation_list) {
                                implementation_list =
                                    existing_implementation_list;
                            }
                        }

                        implementation_list.push(item);
                    }
                }
            }
            some(associated_trait) {
                let def =
                  self.crate_context.tcx.def_map.get(associated_trait.ref_id);
                let def_id = def_id_of_def(def);

                let implementation_list;
                alt self.info.extension_methods.find(def_id) {
                    none {
                        implementation_list = @dvec();
                    }
                    some(existing_implementation_list) {
                        implementation_list = existing_implementation_list;
                    }
                }

                implementation_list.push(item);
            }
        }

        // Add the implementation to the mapping from implementation to base
        // type def ID, if there is a base type for this implementation.
        alt self.get_base_type_def_id(self_type.ty) {
            none {
                // Nothing to do.
            }
            some(base_type_def_id) {
                self.base_type_def_ids.insert(item.id, base_type_def_id);
            }
        }
    }

    fn get_base_type(original_type: t) -> option<t> {
        alt get(original_type).struct {
            ty_box(base_mutability_and_type) |
            ty_uniq(base_mutability_and_type) |
            ty_ptr(base_mutability_and_type) |
            ty_rptr(_, base_mutability_and_type) {
                self.get_base_type(base_mutability_and_type.ty)
            }

            ty_enum(*) | ty_trait(*) | ty_class(*) {
                some(original_type)
            }

            ty_nil | ty_bot | ty_bool | ty_int(*) | ty_uint(*) | ty_float(*) |
            ty_estr(*) | ty_evec(*) | ty_rec(*) |
            ty_fn(*) | ty_tup(*) | ty_var(*) | ty_var_integral(*) |
            ty_param(*) | ty_self | ty_type | ty_opaque_box |
            ty_opaque_closure_ptr(*) | ty_unboxed_vec(*) {
                none
            }
        }
    }

    // Returns the def ID of the base type.
    fn get_base_type_def_id(original_type: t) -> option<def_id> {
        alt self.get_base_type(original_type) {
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
                        fail ~"get_base_type() returned a type that \
                               wasn't an enum, class, or trait";
                    }
                }
            }
        }
    }

    fn check_implementation_coherence(_trait_def_id: def_id,
                                      implementations: @dvec<@item>) {

        // Unify pairs of polytypes.
        for implementations.eachi |i, implementation_a| {
            let polytype_a =
                self.get_self_type_for_implementation(implementation_a);
            for range(i + 1, implementations.len()) |j| {
                let implementation_b = implementations.get_elt(j);
                let polytype_b =
                    self.get_self_type_for_implementation(implementation_b);

                if self.polytypes_unify(polytype_a, polytype_b) {
                    let session = self.crate_context.tcx.sess;
                    session.span_err(implementation_b.span,
                                     ~"conflicting implementations for a \
                                      trait");
                    session.span_note(
                        implementation_a.span,
                        ~"note conflicting implementation here");
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
            if polytype.rp {none}
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

    fn get_self_type_for_implementation(implementation: @item)
                                     -> ty_param_bounds_and_ty {

        alt implementation.node {
            item_impl(*) {
                let def = local_def(implementation.id);
                ret self.crate_context.tcx.tcache.get(def);
            }
            _ {
                self.crate_context.tcx.sess.span_bug(
                    implementation.span,
                    ~"not an implementation");
            }
        }
    }

    // Privileged scope checking

    fn check_privileged_scopes(crate: @crate) {
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
                    item_impl(_, optional_trait_ref, _, _) {
                        alt self.base_type_def_ids.find(item.id) {
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
                                    // if the trait is defined in the same
                                    // crate.

                                    alt optional_trait_ref {
                                        none {
                                            // There is no trait to implement,
                                            // so this is an error.

                                            let session =
                                                self.crate_context.tcx.sess;
                                            session.span_warn(item.span,
                                                              ~"cannot \
                                                               implement \
                                                               inherent \
                                                               methods for a \
                                                               type outside \
                                                               the scope the \
                                                               type was \
                                                               defined in; \
                                                               define and \
                                                               implement a \
                                                               trait \
                                                               instead");
                                        }
                                        some(trait_ref) {
                                            // This is OK if and only if the
                                            // trait was defined in this
                                            // crate.

                                            let def_map = self.crate_context
                                                .tcx.def_map;
                                            let trait_def =
                                                def_map.get(trait_ref.ref_id);
                                            let trait_id =
                                                def_id_of_def(trait_def);
                                            if trait_id.crate != local_crate {
                                                let session = self
                                                    .crate_context.tcx.sess;
                                                session.span_warn(item.span,
                                                                  ~"cannot \
                                                                   provide \
                                                                   an \
                                                                   extension \
                                                                   implement\
                                                                      ation \
                                                                   for a \
                                                                   trait not \
                                                                   defined \
                                                                   in this \
                                                                   crate");
                                            }
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
}

fn check_coherence(crate_context: @crate_ctxt, crate: @crate) {
    CoherenceChecker(crate_context).check_coherence(crate);
}

