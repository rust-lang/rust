// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Give useful errors and suggestions to users when a method can't be
//! found or is otherwise invalid.

use astconv::AstConv;
use check::{self, FnCtxt};
use middle::ty::{self, Ty};
use util::ppaux::UserString;

use syntax::ast;
use syntax::codemap::Span;

use super::{MethodError, CandidateSource, impl_method, trait_method};

pub fn report_error<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                              span: Span,
                              rcvr_ty: Ty<'tcx>,
                              method_name: ast::Name,
                              error: MethodError)
{
    match error {
        MethodError::NoMatch(static_sources) => {
            let cx = fcx.tcx();
            let method_ustring = method_name.user_string(cx);

            // True if the type is a struct and contains a field with
            // the same name as the not-found method
            let is_field = match rcvr_ty.sty {
                ty::ty_struct(did, _) =>
                    ty::lookup_struct_fields(cx, did)
                        .iter()
                        .any(|f| f.name.user_string(cx) == method_ustring),
                _ => false
            };

            fcx.type_error_message(
                span,
                |actual| {
                    format!("type `{}` does not implement any \
                             method in scope named `{}`",
                            actual,
                            method_ustring)
                },
                rcvr_ty,
                None);

            // If the method has the name of a field, give a help note
            if is_field {
                cx.sess.span_note(span,
                    &format!("use `(s.{0})(...)` if you meant to call the \
                            function stored in the `{0}` field", method_ustring)[]);
            }

            if static_sources.len() > 0 {
                fcx.tcx().sess.fileline_note(
                    span,
                    "found defined static methods, maybe a `self` is missing?");

                report_candidates(fcx, span, method_name, static_sources);
            }
        }

        MethodError::Ambiguity(sources) => {
            span_err!(fcx.sess(), span, E0034,
                      "multiple applicable methods in scope");

            report_candidates(fcx, span, method_name, sources);
        }
    }

    fn report_candidates(fcx: &FnCtxt,
                         span: Span,
                         method_name: ast::Name,
                         mut sources: Vec<CandidateSource>) {
        sources.sort();
        sources.dedup();

        for (idx, source) in sources.iter().enumerate() {
            match *source {
                CandidateSource::ImplSource(impl_did) => {
                    // Provide the best span we can. Use the method, if local to crate, else
                    // the impl, if local to crate (method may be defaulted), else the call site.
                    let method = impl_method(fcx.tcx(), impl_did, method_name).unwrap();
                    let impl_span = fcx.tcx().map.def_id_span(impl_did, span);
                    let method_span = fcx.tcx().map.def_id_span(method.def_id, impl_span);

                    let impl_ty = check::impl_self_ty(fcx, span, impl_did).ty;

                    let insertion = match ty::impl_trait_ref(fcx.tcx(), impl_did) {
                        None => format!(""),
                        Some(trait_ref) => format!(" of the trait `{}`",
                                                   ty::item_path_str(fcx.tcx(),
                                                                     trait_ref.def_id)),
                    };

                    span_note!(fcx.sess(), method_span,
                               "candidate #{} is defined in an impl{} for the type `{}`",
                               idx + 1u,
                               insertion,
                               impl_ty.user_string(fcx.tcx()));
                }
                CandidateSource::TraitSource(trait_did) => {
                    let (_, method) = trait_method(fcx.tcx(), trait_did, method_name).unwrap();
                    let method_span = fcx.tcx().map.def_id_span(method.def_id, span);
                    span_note!(fcx.sess(), method_span,
                               "candidate #{} is defined in the trait `{}`",
                               idx + 1u,
                               ty::item_path_str(fcx.tcx(), trait_did));
                }
            }
        }
    }
}
