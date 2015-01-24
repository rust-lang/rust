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

use CrateCtxt;

use astconv::AstConv;
use check::{self, FnCtxt};
use middle::ty::{self, Ty};
use middle::def;
use metadata::{csearch, cstore, decoder};
use util::ppaux::UserString;

use syntax::{ast, ast_util};
use syntax::codemap::Span;

use std::cell;
use std::cmp::Ordering;

use super::{MethodError, CandidateSource, impl_method, trait_method};

pub fn report_error<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                              span: Span,
                              rcvr_ty: Ty<'tcx>,
                              method_name: ast::Name,
                              error: MethodError)
{
    match error {
        MethodError::NoMatch(static_sources, out_of_scope_traits) => {
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

            suggest_traits_to_import(fcx, span, rcvr_ty, method_name, out_of_scope_traits)
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


pub type AllTraitsVec = Vec<TraitInfo>;

fn suggest_traits_to_import<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                      span: Span,
                                      _rcvr_ty: Ty<'tcx>,
                                      method_name: ast::Name,
                                      valid_out_of_scope_traits: Vec<ast::DefId>)
{
    let tcx = fcx.tcx();
    let method_ustring = method_name.user_string(tcx);

    if !valid_out_of_scope_traits.is_empty() {
        let mut candidates = valid_out_of_scope_traits;
        candidates.sort();
        candidates.dedup();
        let msg = format!(
            "methods from traits can only be called if the trait is in scope; \
             the following {traits_are} implemented but not in scope, \
             perhaps add a `use` for {one_of_them}:",
            traits_are = if candidates.len() == 1 {"trait is"} else {"traits are"},
            one_of_them = if candidates.len() == 1 {"it"} else {"one of them"});

        fcx.sess().fileline_help(span, &msg[]);

        for (i, trait_did) in candidates.iter().enumerate() {
            fcx.sess().fileline_help(span,
                                     &*format!("candidate #{}: use `{}`",
                                               i + 1,
                                               ty::item_path_str(fcx.tcx(), *trait_did)))

        }
        return
    }

    // there's no implemented traits, so lets suggest some traits to implement
    let mut candidates = all_traits(fcx.ccx)
        .filter(|info| trait_method(tcx, info.def_id, method_name).is_some())
        .collect::<Vec<_>>();

    if candidates.len() > 0 {
        // sort from most relevant to least relevant
        candidates.sort_by(|a, b| a.cmp(b).reverse());
        candidates.dedup();

        let msg = format!(
            "methods from traits can only be called if the trait is implemented and in scope; \
             the following {traits_define} a method `{name}`, \
             perhaps you need to implement {one_of_them}:",
            traits_define = if candidates.len() == 1 {"trait defines"} else {"traits define"},
            one_of_them = if candidates.len() == 1 {"it"} else {"one of them"},
            name = method_ustring);

        fcx.sess().fileline_help(span, &msg[]);

        for (i, trait_info) in candidates.iter().enumerate() {
            fcx.sess().fileline_help(span,
                                     &*format!("candidate #{}: `{}`",
                                               i + 1,
                                               ty::item_path_str(fcx.tcx(), trait_info.def_id)))
        }
    }
}

#[derive(Copy)]
pub struct TraitInfo {
    pub def_id: ast::DefId,
}

impl TraitInfo {
    fn new(def_id: ast::DefId) -> TraitInfo {
        TraitInfo {
            def_id: def_id,
        }
    }
}
impl PartialEq for TraitInfo {
    fn eq(&self, other: &TraitInfo) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for TraitInfo {}
impl PartialOrd for TraitInfo {
    fn partial_cmp(&self, other: &TraitInfo) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for TraitInfo {
    fn cmp(&self, other: &TraitInfo) -> Ordering {
        // accessible traits are more important/relevant than
        // inaccessible ones, local crates are more important than
        // remote ones (local: cnum == 0), and NodeIds just for
        // totality.

        let lhs = (other.def_id.krate, other.def_id.node);
        let rhs = (self.def_id.krate, self.def_id.node);
        lhs.cmp(&rhs)
    }
}

/// Retrieve all traits in this crate and any dependent crates.
pub fn all_traits<'a>(ccx: &'a CrateCtxt) -> AllTraits<'a> {
    if ccx.all_traits.borrow().is_none() {
        use syntax::visit;

        let mut traits = vec![];

        // Crate-local:
        //
        // meh.
        struct Visitor<'a, 'b: 'a, 'tcx: 'a + 'b> {
            traits: &'a mut AllTraitsVec,
        }
        impl<'v,'a, 'b, 'tcx> visit::Visitor<'v> for Visitor<'a, 'b, 'tcx> {
            fn visit_item(&mut self, i: &'v ast::Item) {
                match i.node {
                    ast::ItemTrait(..) => {
                        self.traits.push(TraitInfo::new(ast_util::local_def(i.id)));
                    }
                    _ => {}
                }
                visit::walk_item(self, i)
            }
        }
        visit::walk_crate(&mut Visitor {
            traits: &mut traits
        }, ccx.tcx.map.krate());

        // Cross-crate:
        fn handle_external_def(traits: &mut AllTraitsVec,
                               ccx: &CrateCtxt,
                               cstore: &cstore::CStore,
                               dl: decoder::DefLike) {
            match dl {
                decoder::DlDef(def::DefTrait(did)) => {
                    traits.push(TraitInfo::new(did));
                }
                decoder::DlDef(def::DefMod(did)) => {
                    csearch::each_child_of_item(cstore, did, |dl, _, _| {
                        handle_external_def(traits, ccx, cstore, dl)
                    })
                }
                _ => {}
            }
        }
        let cstore = &ccx.tcx.sess.cstore;
        cstore.iter_crate_data(|&mut: cnum, _| {
            csearch::each_top_level_item_of_crate(cstore, cnum, |dl, _, _| {
                handle_external_def(&mut traits, ccx, cstore, dl)
            })
        });

        *ccx.all_traits.borrow_mut() = Some(traits);
    }

    let borrow = ccx.all_traits.borrow();
    assert!(borrow.is_some());
    AllTraits {
        borrow: borrow,
        idx: 0
    }
}

pub struct AllTraits<'a> {
    borrow: cell::Ref<'a Option<AllTraitsVec>>,
    idx: usize
}

impl<'a> Iterator for AllTraits<'a> {
    type Item = TraitInfo;

    fn next(&mut self) -> Option<TraitInfo> {
        let AllTraits { ref borrow, ref mut idx } = *self;
        // ugh.
        borrow.as_ref().unwrap().get(*idx).map(|info| {
            *idx += 1;
            *info
        })
    }
}
