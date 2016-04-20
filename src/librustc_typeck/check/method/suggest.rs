// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Give useful errors and suggestions to users when an item can't be
//! found or is otherwise invalid.

use CrateCtxt;

use astconv::AstConv;
use check::{self, FnCtxt, UnresolvedTypeAction, autoderef};
use rustc::hir::map as hir_map;
use rustc::ty::{self, Ty, ToPolyTraitRef, ToPredicate, TypeFoldable};
use middle::cstore::{self, CrateStore};
use hir::def::Def;
use hir::def_id::DefId;
use middle::lang_items::FnOnceTraitLangItem;
use rustc::ty::subst::Substs;
use rustc::ty::LvaluePreference;
use rustc::traits::{Obligation, SelectionContext};
use util::nodemap::{FnvHashSet};


use syntax::ast;
use syntax::codemap::Span;
use syntax::errors::DiagnosticBuilder;
use rustc::hir::print as pprust;
use rustc::hir;
use rustc::hir::Expr_;

use std::cell;
use std::cmp::Ordering;

use super::{MethodError, NoMatchData, CandidateSource, impl_item, trait_item};
use super::probe::Mode;

fn is_fn_ty<'a, 'tcx>(ty: &Ty<'tcx>, fcx: &FnCtxt<'a, 'tcx>, span: Span) -> bool {
    let cx = fcx.tcx();
    match ty.sty {
        // Not all of these (e.g. unsafe fns) implement FnOnce
        // so we look for these beforehand
        ty::TyClosure(..) | ty::TyFnDef(..) | ty::TyFnPtr(_) => true,
        // If it's not a simple function, look for things which implement FnOnce
        _ => {
            if let Ok(fn_once_trait_did) =
                    cx.lang_items.require(FnOnceTraitLangItem) {
                let infcx = fcx.infcx();
                let (_, _, opt_is_fn) = autoderef(fcx,
                                                  span,
                                                  ty,
                                                  || None,
                                                  UnresolvedTypeAction::Ignore,
                                                  LvaluePreference::NoPreference,
                                                  |ty, _| {
                    infcx.probe(|_| {
                        let fn_once_substs =
                            Substs::new_trait(vec![infcx.next_ty_var()],
                                              Vec::new(),
                                              ty);
                        let trait_ref =
                          ty::TraitRef::new(fn_once_trait_did,
                                            cx.mk_substs(fn_once_substs));
                        let poly_trait_ref = trait_ref.to_poly_trait_ref();
                        let obligation = Obligation::misc(span,
                                                          fcx.body_id,
                                                          poly_trait_ref
                                                             .to_predicate());
                        let mut selcx = SelectionContext::new(infcx);

                        if selcx.evaluate_obligation(&obligation) {
                            Some(())
                        } else {
                            None
                        }
                    })
                });

                opt_is_fn.is_some()
            } else {
                false
            }
        }
    }
}

pub fn report_error<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                              span: Span,
                              rcvr_ty: Ty<'tcx>,
                              item_name: ast::Name,
                              rcvr_expr: Option<&hir::Expr>,
                              error: MethodError<'tcx>)
{
    // avoid suggestions when we don't know what's going on.
    if rcvr_ty.references_error() {
        return
    }

    match error {
        MethodError::NoMatch(NoMatchData { static_candidates: static_sources,
                                           unsatisfied_predicates,
                                           out_of_scope_traits,
                                           mode, .. }) => {
            let cx = fcx.tcx();

            let mut err = fcx.type_error_struct(
                span,
                |actual| {
                    format!("no {} named `{}` found for type `{}` \
                             in the current scope",
                            if mode == Mode::MethodCall { "method" }
                            else { "associated item" },
                            item_name,
                            actual)
                },
                rcvr_ty,
                None);

            // If the item has the name of a field, give a help note
            if let (&ty::TyStruct(def, substs), Some(expr)) = (&rcvr_ty.sty, rcvr_expr) {
                if let Some(field) = def.struct_variant().find_field_named(item_name) {
                    let expr_string = match cx.sess.codemap().span_to_snippet(expr.span) {
                        Ok(expr_string) => expr_string,
                        _ => "s".into() // Default to a generic placeholder for the
                                        // expression when we can't generate a string
                                        // snippet
                    };

                    let field_ty = field.ty(cx, substs);

                    if is_fn_ty(&field_ty, &fcx, span) {
                        err.span_note(span,
                                      &format!("use `({0}.{1})(...)` if you meant to call \
                                               the function stored in the `{1}` field",
                                               expr_string, item_name));
                    } else {
                        err.span_note(span, &format!("did you mean to write `{0}.{1}`?",
                                                     expr_string, item_name));
                    }
                }
            }

            if is_fn_ty(&rcvr_ty, &fcx, span) {
                macro_rules! report_function {
                    ($span:expr, $name:expr) => {
                        err.note(&format!("{} is a function, perhaps you wish to call it",
                                     $name));
                    }
                }

                if let Some(expr) = rcvr_expr {
                    if let Ok (expr_string) = cx.sess.codemap().span_to_snippet(expr.span) {
                        report_function!(expr.span, expr_string);
                        err.span_suggestion(expr.span,
                                            "try calling the base function:",
                                            format!("{}()",
                                                    expr_string));
                    }
                    else if let Expr_::ExprPath(_, path) = expr.node.clone() {
                        if let Some(segment) = path.segments.last() {
                            report_function!(expr.span, segment.identifier.name);
                        }
                    }
                }
            }

            if !static_sources.is_empty() {
                err.note(
                    "found the following associated functions; to be used as \
                     methods, functions must have a `self` parameter");

                report_candidates(fcx, &mut err, span, item_name, static_sources);
            }

            if !unsatisfied_predicates.is_empty() {
                let bound_list = unsatisfied_predicates.iter()
                    .map(|p| format!("`{} : {}`",
                                     p.self_ty(),
                                     p))
                    .collect::<Vec<_>>()
                    .join(", ");
                err.note(
                    &format!("the method `{}` exists but the \
                             following trait bounds were not satisfied: {}",
                             item_name,
                             bound_list));
            }

            suggest_traits_to_import(fcx, &mut err, span, rcvr_ty, item_name,
                                     rcvr_expr, out_of_scope_traits);
            err.emit();
        }

        MethodError::Ambiguity(sources) => {
            let mut err = struct_span_err!(fcx.sess(), span, E0034,
                                           "multiple applicable items in scope");

            report_candidates(fcx, &mut err, span, item_name, sources);
            err.emit();
        }

        MethodError::ClosureAmbiguity(trait_def_id) => {
            let msg = format!("the `{}` method from the `{}` trait cannot be explicitly \
                               invoked on this closure as we have not yet inferred what \
                               kind of closure it is",
                               item_name,
                               fcx.tcx().item_path_str(trait_def_id));
            let msg = if let Some(callee) = rcvr_expr {
                format!("{}; use overloaded call notation instead (e.g., `{}()`)",
                        msg, pprust::expr_to_string(callee))
            } else {
                msg
            };
            fcx.sess().span_err(span, &msg);
        }

        MethodError::PrivateMatch(def) => {
            let msg = format!("{} `{}` is private", def.kind_name(), item_name);
            fcx.tcx().sess.span_err(span, &msg);
        }
    }

    fn report_candidates(fcx: &FnCtxt,
                         err: &mut DiagnosticBuilder,
                         span: Span,
                         item_name: ast::Name,
                         mut sources: Vec<CandidateSource>) {
        sources.sort();
        sources.dedup();

        for (idx, source) in sources.iter().enumerate() {
            match *source {
                CandidateSource::ImplSource(impl_did) => {
                    // Provide the best span we can. Use the item, if local to crate, else
                    // the impl, if local to crate (item may be defaulted), else the call site.
                    let item = impl_item(fcx.tcx(), impl_did, item_name)
                        .or_else(|| {
                            trait_item(
                                fcx.tcx(),
                                fcx.tcx().impl_trait_ref(impl_did).unwrap().def_id,
                                item_name
                            )
                        }).unwrap();
                    let impl_span = fcx.tcx().map.def_id_span(impl_did, span);
                    let item_span = fcx.tcx().map.def_id_span(item.def_id(), impl_span);

                    let impl_ty = check::impl_self_ty(fcx, span, impl_did).ty;

                    let insertion = match fcx.tcx().impl_trait_ref(impl_did) {
                        None => format!(""),
                        Some(trait_ref) => {
                            format!(" of the trait `{}`",
                                    fcx.tcx().item_path_str(trait_ref.def_id))
                        }
                    };

                    span_note!(err, item_span,
                               "candidate #{} is defined in an impl{} for the type `{}`",
                               idx + 1,
                               insertion,
                               impl_ty);
                }
                CandidateSource::TraitSource(trait_did) => {
                    let item = trait_item(fcx.tcx(), trait_did, item_name).unwrap();
                    let item_span = fcx.tcx().map.def_id_span(item.def_id(), span);
                    span_note!(err, item_span,
                               "candidate #{} is defined in the trait `{}`",
                               idx + 1,
                               fcx.tcx().item_path_str(trait_did));
                }
            }
        }
    }
}


pub type AllTraitsVec = Vec<TraitInfo>;

fn suggest_traits_to_import<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                      err: &mut DiagnosticBuilder,
                                      span: Span,
                                      rcvr_ty: Ty<'tcx>,
                                      item_name: ast::Name,
                                      rcvr_expr: Option<&hir::Expr>,
                                      valid_out_of_scope_traits: Vec<DefId>)
{
    let tcx = fcx.tcx();

    if !valid_out_of_scope_traits.is_empty() {
        let mut candidates = valid_out_of_scope_traits;
        candidates.sort();
        candidates.dedup();
        let msg = format!(
            "items from traits can only be used if the trait is in scope; \
             the following {traits_are} implemented but not in scope, \
             perhaps add a `use` for {one_of_them}:",
            traits_are = if candidates.len() == 1 {"trait is"} else {"traits are"},
            one_of_them = if candidates.len() == 1 {"it"} else {"one of them"});

        err.help(&msg[..]);

        for (i, trait_did) in candidates.iter().enumerate() {
            err.help(&format!("candidate #{}: `use {}`",
                              i + 1,
                              fcx.tcx().item_path_str(*trait_did)));
        }
        return
    }

    let type_is_local = type_derefs_to_local(fcx, span, rcvr_ty, rcvr_expr);

    // there's no implemented traits, so lets suggest some traits to
    // implement, by finding ones that have the item name, and are
    // legal to implement.
    let mut candidates = all_traits(fcx.ccx)
        .filter(|info| {
            // we approximate the coherence rules to only suggest
            // traits that are legal to implement by requiring that
            // either the type or trait is local. Multidispatch means
            // this isn't perfect (that is, there are cases when
            // implementing a trait would be legal but is rejected
            // here).
            (type_is_local || info.def_id.is_local())
                && trait_item(tcx, info.def_id, item_name).is_some()
        })
        .collect::<Vec<_>>();

    if !candidates.is_empty() {
        // sort from most relevant to least relevant
        candidates.sort_by(|a, b| a.cmp(b).reverse());
        candidates.dedup();

        // FIXME #21673 this help message could be tuned to the case
        // of a type parameter: suggest adding a trait bound rather
        // than implementing.
        let msg = format!(
            "items from traits can only be used if the trait is implemented and in scope; \
             the following {traits_define} an item `{name}`, \
             perhaps you need to implement {one_of_them}:",
            traits_define = if candidates.len() == 1 {"trait defines"} else {"traits define"},
            one_of_them = if candidates.len() == 1 {"it"} else {"one of them"},
            name = item_name);

        err.help(&msg[..]);

        for (i, trait_info) in candidates.iter().enumerate() {
            err.help(&format!("candidate #{}: `{}`",
                              i + 1,
                              fcx.tcx().item_path_str(trait_info.def_id)));
        }
    }
}

/// Checks whether there is a local type somewhere in the chain of
/// autoderefs of `rcvr_ty`.
fn type_derefs_to_local<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                  span: Span,
                                  rcvr_ty: Ty<'tcx>,
                                  rcvr_expr: Option<&hir::Expr>) -> bool {
    fn is_local(ty: Ty) -> bool {
        match ty.sty {
            ty::TyEnum(def, _) | ty::TyStruct(def, _) => def.did.is_local(),

            ty::TyTrait(ref tr) => tr.principal_def_id().is_local(),

            ty::TyParam(_) => true,

            // everything else (primitive types etc.) is effectively
            // non-local (there are "edge" cases, e.g. (LocalType,), but
            // the noise from these sort of types is usually just really
            // annoying, rather than any sort of help).
            _ => false
        }
    }

    // This occurs for UFCS desugaring of `T::method`, where there is no
    // receiver expression for the method call, and thus no autoderef.
    if rcvr_expr.is_none() {
        return is_local(fcx.resolve_type_vars_if_possible(rcvr_ty));
    }

    check::autoderef(fcx, span, rcvr_ty, || None,
                     check::UnresolvedTypeAction::Ignore, ty::NoPreference,
                     |ty, _| {
        if is_local(ty) {
            Some(())
        } else {
            None
        }
    }).2.is_some()
}

#[derive(Copy, Clone)]
pub struct TraitInfo {
    pub def_id: DefId,
}

impl TraitInfo {
    fn new(def_id: DefId) -> TraitInfo {
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
        // local crates are more important than remote ones (local:
        // cnum == 0), and otherwise we throw in the defid for totality

        let lhs = (other.def_id.krate, other.def_id);
        let rhs = (self.def_id.krate, self.def_id);
        lhs.cmp(&rhs)
    }
}

/// Retrieve all traits in this crate and any dependent crates.
pub fn all_traits<'a>(ccx: &'a CrateCtxt) -> AllTraits<'a> {
    if ccx.all_traits.borrow().is_none() {
        use rustc::hir::intravisit;

        let mut traits = vec![];

        // Crate-local:
        //
        // meh.
        struct Visitor<'a, 'tcx:'a> {
            map: &'a hir_map::Map<'tcx>,
            traits: &'a mut AllTraitsVec,
        }
        impl<'v, 'a, 'tcx> intravisit::Visitor<'v> for Visitor<'a, 'tcx> {
            fn visit_item(&mut self, i: &'v hir::Item) {
                match i.node {
                    hir::ItemTrait(..) => {
                        let def_id = self.map.local_def_id(i.id);
                        self.traits.push(TraitInfo::new(def_id));
                    }
                    _ => {}
                }
            }
        }
        ccx.tcx.map.krate().visit_all_items(&mut Visitor {
            map: &ccx.tcx.map,
            traits: &mut traits
        });

        // Cross-crate:
        let mut external_mods = FnvHashSet();
        fn handle_external_def(traits: &mut AllTraitsVec,
                               external_mods: &mut FnvHashSet<DefId>,
                               ccx: &CrateCtxt,
                               cstore: &for<'a> cstore::CrateStore<'a>,
                               dl: cstore::DefLike) {
            match dl {
                cstore::DlDef(Def::Trait(did)) => {
                    traits.push(TraitInfo::new(did));
                }
                cstore::DlDef(Def::Mod(did)) => {
                    if !external_mods.insert(did) {
                        return;
                    }
                    for child in cstore.item_children(did) {
                        handle_external_def(traits, external_mods,
                                            ccx, cstore, child.def)
                    }
                }
                _ => {}
            }
        }
        let cstore = &*ccx.tcx.sess.cstore;

        for cnum in ccx.tcx.sess.cstore.crates() {
            for child in cstore.crate_top_level_items(cnum) {
                handle_external_def(&mut traits, &mut external_mods,
                                    ccx, cstore, child.def)
            }
        }

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
    borrow: cell::Ref<'a, Option<AllTraitsVec>>,
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
