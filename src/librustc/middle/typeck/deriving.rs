// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Deriving phase
//
// The purpose of the deriving phase of typechecking is to ensure that, for
// each automatically derived implementation of an automatically-derivable
// trait (for example, Eq), all the subcomponents of the type in question
// also implement the trait. This phase runs after coherence.

use syntax::ast::crate;
use syntax::ast::{def_id, ident};
use syntax::ast::item_impl;
use syntax::ast::node_id;
use syntax::ast::self_ty_;
use syntax::ast::trait_ref;
use syntax::ast_util::{def_id_of_def, dummy_sp};
use syntax::codemap::span;
use syntax::print::pprust;
use syntax::visit::{default_simple_visitor, mk_simple_visitor, visit_crate};
use middle::resolve::{Impl, MethodInfo};
use middle::ty;
use middle::ty::{DerivedFieldInfo, ReVar, re_infer, re_static, substs};
use middle::ty::{ty_class, ty_enum, ty_param_bounds_and_ty};
use /*middle::typeck::*/check::method;
use /*middle::typeck::*/check::vtable;
use /*middle::typeck::*/infer::infer_ctxt;
use /*middle::typeck::*/vtable::{LocationInfo, VtableContext};
use util::ppaux;

struct MethodMatch {
    method_def_id: def_id,
    type_parameter_substitutions: @~[ty::t],
    vtable_result: Option<vtable_res>
}

struct DerivingChecker {
    crate_context: @crate_ctxt
}

fn DerivingChecker_new(crate_context: @crate_ctxt) -> DerivingChecker {
    DerivingChecker {
        crate_context: crate_context,
    }
}

struct TyParamSubstsAndVtableResult {
    type_parameter_substitutions: @~[ty::t],
    vtable_result: Option<vtable_res>
}

impl DerivingChecker {
    /// Matches one substructure type against an implementation.
    fn match_impl_method(impl_info: @Impl,
                         substructure_type: ty::t,
                         method_info: @MethodInfo,
                         span: span) ->
                         Option<TyParamSubstsAndVtableResult> {
        let tcx = self.crate_context.tcx;

        let impl_self_tpbt = ty::lookup_item_type(tcx, impl_info.did);

        let inference_context = infer::new_infer_ctxt(self.crate_context.tcx);
        let region = inference_context.next_region_var_nb(span);
        let transformed_type = method::transform_self_type_for_method(
            tcx, Some(region), impl_self_tpbt.ty, method_info.self_type);

        let substs = {
            self_r: None,
            self_ty: None,
            tps: inference_context.next_ty_vars(impl_self_tpbt.bounds.len())
        };
        let transformed_type = ty::subst(
            self.crate_context.tcx, &substs, transformed_type);

        // Automatically reference the substructure type.
        let region = inference_context.next_region_var_nb(span);
        let substructure_type = ty::mk_rptr(
            self.crate_context.tcx,
            region,
            { ty: substructure_type, mutbl: ast::m_imm });

        debug!("(matching impl method) substructure type %s, transformed \
                type %s, subst tps %u",
               ppaux::ty_to_str(self.crate_context.tcx, substructure_type),
               ppaux::ty_to_str(self.crate_context.tcx, transformed_type),
               substs.tps.len());

        if !infer::mk_subty(inference_context,
                            true,
                            ast_util::dummy_sp(),
                            substructure_type,
                            transformed_type).is_ok() {
            return None;
        }

        // Get the vtables.
        let vtable_result;
        if substs.tps.len() == 0 {
            vtable_result = None;
        } else {
            let vcx = VtableContext {
                ccx: self.crate_context,
                infcx: inference_context
            };
            let location_info = LocationInfo {
                span: span,
                id: impl_info.did.node
            };
            vtable_result = Some(vtable::lookup_vtables(&vcx,
                                                        &location_info,
                                                        impl_self_tpbt.bounds,
                                                        &substs,
                                                        false,
                                                        false));
        }

        // Extract the type parameter substitutions.
        let type_parameter_substitutions = @substs.tps.map(|ty_var|
            inference_context.resolve_type_vars_if_possible(*ty_var));

        Some(TyParamSubstsAndVtableResult {
            type_parameter_substitutions: type_parameter_substitutions,
            vtable_result: vtable_result
        })
    }

    fn check_deriving_for_substructure_type(substructure_type: ty::t,
                                            trait_ref: @trait_ref,
                                            impl_span: span) ->
                                            Option<MethodMatch> {
        let tcx = self.crate_context.tcx;
        let sess = tcx.sess;
        let coherence_info = self.crate_context.coherence_info;
        let trait_id = def_id_of_def(tcx.def_map.get(trait_ref.ref_id));
        match coherence_info.extension_methods.find(trait_id) {
            None => {
                sess.span_bug(impl_span, ~"no extension method info found \
                                           for this trait");
            }
            Some(impls) => {
                // Try to unify each of these impls with the substructure
                // type.
                //
                // NB: Using range to avoid a recursive-use-of-dvec error.
                for uint::range(0, impls.len()) |i| {
                    let impl_info = impls[i];
                    for uint::range(0, impl_info.methods.len()) |j| {
                        let method_info = impl_info.methods[j];
                        match self.match_impl_method(impl_info,
                                                     substructure_type,
                                                     method_info,
                                                     trait_ref.path.span) {
                            Some(move result) => {
                                return Some(MethodMatch {
                                    method_def_id: method_info.did,
                                    type_parameter_substitutions:
                                        result.type_parameter_substitutions,
                                    vtable_result: result.vtable_result
                                });
                            }
                            None => {}  // Continue.
                        }
                    }
                }
            }
        }
        return None;
    }

    fn check_deriving_for_struct(struct_def_id: def_id,
                                 struct_substs: &substs,
                                 trait_ref: @trait_ref,
                                 impl_id: node_id,
                                 impl_span: span) {
        let tcx = self.crate_context.tcx;
        let field_info = dvec::DVec();
        for ty::lookup_class_fields(tcx, struct_def_id).each |field| {
            let field_type = ty::lookup_field_type(
                tcx, struct_def_id, field.id, struct_substs);
            match self.check_deriving_for_substructure_type(field_type,
                                                            trait_ref,
                                                            impl_span) {
                Some(method_match) => {
                    field_info.push(DerivedFieldInfo {
                        method_origin:
                            method_static(method_match.method_def_id),
                        type_parameter_substitutions:
                            method_match.type_parameter_substitutions,
                        vtable_result:
                            method_match.vtable_result
                    });
                }
                None => {
                    let trait_str = pprust::path_to_str(
                        trait_ref.path, tcx.sess.parse_sess.interner);
                    tcx.sess.span_err(impl_span,
                                      fmt!("cannot automatically derive an \
                                            implementation for `%s`: field \
                                            `%s` does not implement the \
                                            trait `%s`",
                                           trait_str,
                                           tcx.sess.str_of(field.ident),
                                           trait_str));
                }
            }
        }

        let field_info = @dvec::unwrap(move field_info);
        tcx.deriving_struct_methods.insert(local_def(impl_id), field_info);
    }

    fn check_deriving_for_enum(enum_def_id: def_id,
                               enum_substs: &substs,
                               trait_ref: @trait_ref,
                               impl_id: node_id,
                               impl_span: span) {
        let tcx = self.crate_context.tcx;
        let enum_methods = dvec::DVec();
        let variants = ty::substd_enum_variants(
            tcx, enum_def_id, enum_substs);
        for variants.each |enum_variant_info| {
            let variant_methods = dvec::DVec();
            for enum_variant_info.args.eachi |i, variant_arg_type| {
                match self.check_deriving_for_substructure_type(
                        *variant_arg_type, trait_ref, impl_span) {
                    Some(method_match) => {
                        variant_methods.push(DerivedFieldInfo {
                            method_origin:
                                method_static(method_match.method_def_id),
                            type_parameter_substitutions:
                                method_match.type_parameter_substitutions,
                            vtable_result:
                                method_match.vtable_result
                        });
                    }
                    None => {
                        let trait_str = pprust::path_to_str(
                            trait_ref.path, tcx.sess.parse_sess.interner);
                        tcx.sess.span_err(impl_span,
                                          fmt!("cannot automatically derive \
                                                an implementation for `%s`: \
                                                argument %u of variant `%s` \
                                                does not implement the trait \
                                                `%s`",
                                               trait_str,
                                               i + 1,
                                               tcx.sess.str_of(
                                                    enum_variant_info.name),
                                               trait_str));
                    }
                }
            }
            enum_methods.push(@dvec::unwrap(move variant_methods));
        }

        let enum_methods = @dvec::unwrap(move enum_methods);
        tcx.deriving_enum_methods.insert(local_def(impl_id), enum_methods);
    }

    fn check_deriving(crate: @crate) {
        let tcx = self.crate_context.tcx;
        visit_crate(*crate, (), mk_simple_visitor(@{
            visit_item: |item| {
                match item.node {
                    item_impl(_, Some(trait_ref), _, _) => {
                        // Determine whether there were any automatically-
                        // derived methods in this implementation.
                        let impl_did = local_def(item.id);
                        if tcx.automatically_derived_methods_for_impl.
                                contains_key(impl_did) {
                            // XXX: This does not handle generic impls.
                            let superty = ty::lookup_item_type(
                                tcx, local_def(item.id)).ty;
                            match ty::get(superty).sty {
                                ty_enum(def_id, ref substs) => {
                                    self.check_deriving_for_enum(
                                        def_id,
                                        substs,
                                        trait_ref,
                                        item.id,
                                        item.span);
                                }
                                ty_class(def_id, ref substs) => {
                                    self.check_deriving_for_struct(
                                        def_id,
                                        substs,
                                        trait_ref,
                                        item.id,
                                        item.span);
                                }
                                _ => {
                                    tcx.sess.span_err(item.span,
                                                      ~"only enums and \
                                                        structs may have \
                                                        implementations \
                                                        automatically \
                                                        derived for them");
                                }
                            }
                        }
                    }
                    _ => {}
                }
            },
            ..*default_simple_visitor()
        }));
    }
}

pub fn check_deriving(crate_context: @crate_ctxt, crate: @crate) {
    let deriving_checker = @DerivingChecker_new(crate_context);
    deriving_checker.check_deriving(crate);
}

