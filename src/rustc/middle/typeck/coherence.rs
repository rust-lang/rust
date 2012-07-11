// Coherence phase
//
// The job of the coherence phase of typechecking is to ensure that each trait
// has at most one implementation for each type. Then we build a mapping from
// each trait in the system to its implementations.

import middle::ty::{get, t, ty_box, ty_uniq, ty_ptr, ty_rptr, ty_enum};
import middle::ty::{ty_class, ty_nil, ty_bot, ty_bool, ty_int, ty_uint};
import middle::ty::{ty_float, ty_str, ty_estr, ty_vec, ty_evec, ty_rec};
import middle::ty::{ty_fn, ty_trait, ty_tup, ty_var, ty_var_integral};
import middle::ty::{ty_param, ty_self, ty_constr, ty_type, ty_opaque_box};
import middle::ty::{ty_opaque_closure_ptr, ty_unboxed_vec, new_ty_hash};
import middle::ty::{subst};
import middle::typeck::infer::{infer_ctxt, mk_eqty, new_infer_ctxt};
import syntax::ast::{crate, def_id, item, item_impl, method, region_param};
import syntax::ast::{trait_ref};
import syntax::ast_util::{def_id_of_def, new_def_hash};
import syntax::visit::{default_simple_visitor, mk_simple_visitor};
import syntax::visit::{visit_crate};

import dvec::{dvec, extensions};
import result::{extensions};
import std::map::hashmap;
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

    new(crate_context: @crate_ctxt) {
        self.crate_context = crate_context;
        self.inference_context = new_infer_ctxt(crate_context.tcx);
        self.info = @CoherenceInfo();
    }

    fn check_coherence(crate: @crate) {
        // Check implementations. This populates the tables containing the
        // inherent methods and extension methods.

        visit_crate(*crate, (), mk_simple_visitor(@{
            visit_item: |item| {
                alt item.node {
                    item_impl(_, _, associated_trait, self_type, _) {
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
                                          "no base type found for inherent \
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
                    self.crate_context.tcx.def_map.get(associated_trait.id);
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
    }

    fn get_base_type(original_type: t) -> option<t> {
        alt get(original_type).struct {
            ty_box(base_mutability_and_type) |
            ty_uniq(base_mutability_and_type) |
            ty_ptr(base_mutability_and_type) |
            ty_rptr(_, base_mutability_and_type) {
                self.get_base_type(base_mutability_and_type.ty)
            }

            ty_enum(*) | ty_class(*) {
                some(original_type)
            }

            ty_nil | ty_bot | ty_bool | ty_int(*) | ty_uint(*) | ty_float(*) |
            ty_str | ty_estr(*) | ty_vec(*) | ty_evec(*) | ty_rec(*) |
            ty_fn(*) | ty_trait(*) | ty_tup(*) | ty_var(*) |
            ty_var_integral(*) | ty_param(*) | ty_self | ty_constr(*) |
            ty_type | ty_opaque_box | ty_opaque_closure_ptr(*) |
            ty_unboxed_vec(*) {
                none
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
                                     "conflicting implementations for a \
                                      trait");
                    session.span_note(implementation_a.span,
                                      "note conflicting implementation here");
                }
            }
        }
    }

    fn polytypes_unify(polytype_a: ty_param_bounds_and_ty,
                       polytype_b: ty_param_bounds_and_ty)
                    -> bool {

        let monotype_a = self.universally_quantify_polytype(polytype_a);
        let monotype_b = self.universally_quantify_polytype(polytype_b);
        ret mk_eqty(self.inference_context, monotype_a, monotype_b).is_ok();
    }

    // Converts a polytype to a monotype by replacing all parameters with
    // type variables.
    fn universally_quantify_polytype(polytype: ty_param_bounds_and_ty) -> t {
        let self_region;
        alt polytype.rp {
            ast::rp_none {
                self_region = none;
            }
            ast::rp_self {
                self_region = some(self.inference_context.next_region_var())
            }
        };

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
                self.crate_context.tcx.sess.span_bug(implementation.span,
                                                     "not an implementation");
            }
        }
    }
}

fn check_coherence(crate_context: @crate_ctxt, crate: @crate) {
    CoherenceChecker(crate_context).check_coherence(crate);
}

