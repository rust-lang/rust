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
use syntax::ast_util::def_id_of_def;
use syntax::codemap::span;
use syntax::print::pprust;
use syntax::visit::{default_simple_visitor, mk_simple_visitor, visit_crate};
use middle::resolve::{Impl, MethodInfo};
use middle::ty;
use middle::ty::{substs, ty_class, ty_enum, ty_param_bounds_and_ty};
use /*middle::typeck::*/check::method;
use /*middle::typeck::*/infer::infer_ctxt;

struct DerivingChecker {
    crate_context: @crate_ctxt,
    inference_context: infer_ctxt
}

fn DerivingChecker_new(crate_context: @crate_ctxt) -> DerivingChecker {
    DerivingChecker {
        crate_context: crate_context,
        inference_context: infer::new_infer_ctxt(crate_context.tcx)
    }
}

impl DerivingChecker {
    /// Matches one substructure type against an implementation.
    fn match_impl_method(impl_info: @Impl,
                         substructure_type: ty::t,
                         method_info: @MethodInfo) -> bool {
        // XXX: Generics and regions are not handled properly.
        let tcx = self.crate_context.tcx;
        let impl_self_ty = ty::lookup_item_type(tcx, impl_info.did).ty;
        let transformed_type = method::transform_self_type_for_method(
            tcx, None, impl_self_ty, method_info.self_type);
        return infer::can_mk_subty(self.inference_context,
                                   substructure_type,
                                   transformed_type).is_ok();
    }

    fn check_deriving_for_substructure_type(substructure_type: ty::t,
                                            trait_ref: @trait_ref,
                                            impl_span: span) ->
                                            Option<def_id> {
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
                for impls.each |impl_info| {
                    for impl_info.methods.each |method_info| {
                        if self.match_impl_method(*impl_info,
                                                  substructure_type,
                                                  *method_info) {
                            return Some(method_info.did);
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
                Some(method_target_def_id) => {
                    field_info.push(method_static(method_target_def_id));
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

    fn check_deriving(crate: @crate) {
        let tcx = self.crate_context.tcx;
        visit_crate(*crate, (), mk_simple_visitor(@{
            visit_item: |item| {
                match item.node {
                    item_impl(_, Some(trait_ref), _, None) => {
                        // XXX: This does not handle generic impls.
                        let superty = ty::lookup_item_type(
                            tcx, local_def(item.id)).ty;
                        match ty::get(superty).sty {
                            ty_enum(_def_id, _substs) => {
                                // XXX: Handle enums.
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
                                                  ~"only enums and structs \
                                                    may have implementations \
                                                    automatically derived \
                                                    for them");
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

