// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Error Reporting Code for the inference engine
//!
//! Because of the way inference, and in particular region inference,
//! works, it often happens that errors are not detected until far after
//! the relevant line of code has been type-checked. Therefore, there is
//! an elaborate system to track why a particular constraint in the
//! inference graph arose so that we can explain to the user what gave
//! rise to a particular error.
//!
//! The basis of the system are the "origin" types. An "origin" is the
//! reason that a constraint or inference variable arose. There are
//! different "origin" enums for different kinds of constraints/variables
//! (e.g., `TypeOrigin`, `RegionVariableOrigin`). An origin always has
//! a span, but also more information so that we can generate a meaningful
//! error message.
//!
//! Having a catalog of all the different reasons an error can arise is
//! also useful for other reasons, like cross-referencing FAQs etc, though
//! we are not really taking advantage of this yet.
//!
//! # Region Inference
//!
//! Region inference is particularly tricky because it always succeeds "in
//! the moment" and simply registers a constraint. Then, at the end, we
//! can compute the full graph and report errors, so we need to be able to
//! store and later report what gave rise to the conflicting constraints.
//!
//! # Subtype Trace
//!
//! Determining whether `T1 <: T2` often involves a number of subtypes and
//! subconstraints along the way. A "TypeTrace" is an extended version
//! of an origin that traces the types and other values that were being
//! compared. It is not necessarily comprehensive (in fact, at the time of
//! this writing it only tracks the root values being compared) but I'd
//! like to extend it to include significant "waypoints". For example, if
//! you are comparing `(T1, T2) <: (T3, T4)`, and the problem is that `T2
//! <: T4` fails, I'd like the trace to include enough information to say
//! "in the 2nd element of the tuple". Similarly, failures when comparing
//! arguments or return types in fn types should be able to cite the
//! specific position, etc.
//!
//! # Reality vs plan
//!
//! Of course, there is still a LOT of code in typeck that has yet to be
//! ported to this system, and which relies on string concatenation at the
//! time of error detection.

use infer;
use super::{InferCtxt, RegionVariableOrigin, SubregionOrigin, TypeTrace, ValuePairs};
use super::region_constraints::GenericKind;
use super::lexical_region_resolve::RegionResolutionError;

use std::fmt;
use hir;
use hir::map as hir_map;
use hir::def_id::DefId;
use middle::region;
use traits::{ObligationCause, ObligationCauseCode};
use ty::{self, Region, Ty, TyCtxt, TypeFoldable, TypeVariants};
use ty::error::TypeError;
use syntax::ast::DUMMY_NODE_ID;
use syntax_pos::{Pos, Span};
use errors::{Applicability, DiagnosticBuilder, DiagnosticStyledString};

use rustc_data_structures::indexed_vec::Idx;

mod note;

mod need_type_info;

pub mod nice_region_error;

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn note_and_explain_region(
        self,
        region_scope_tree: &region::ScopeTree,
        err: &mut DiagnosticBuilder,
        prefix: &str,
        region: ty::Region<'tcx>,
        suffix: &str,
    ) {
        let (description, span) = match *region {
            ty::ReScope(scope) => {
                let new_string;
                let unknown_scope = || {
                    format!(
                        "{}unknown scope: {:?}{}.  Please report a bug.",
                        prefix, scope, suffix
                    )
                };
                let span = scope.span(self, region_scope_tree);
                let tag = match self.hir.find(scope.node_id(self, region_scope_tree)) {
                    Some(hir_map::NodeBlock(_)) => "block",
                    Some(hir_map::NodeExpr(expr)) => match expr.node {
                        hir::ExprCall(..) => "call",
                        hir::ExprMethodCall(..) => "method call",
                        hir::ExprMatch(.., hir::MatchSource::IfLetDesugar { .. }) => "if let",
                        hir::ExprMatch(.., hir::MatchSource::WhileLetDesugar) => "while let",
                        hir::ExprMatch(.., hir::MatchSource::ForLoopDesugar) => "for",
                        hir::ExprMatch(..) => "match",
                        _ => "expression",
                    },
                    Some(hir_map::NodeStmt(_)) => "statement",
                    Some(hir_map::NodeItem(it)) => Self::item_scope_tag(&it),
                    Some(hir_map::NodeTraitItem(it)) => Self::trait_item_scope_tag(&it),
                    Some(hir_map::NodeImplItem(it)) => Self::impl_item_scope_tag(&it),
                    Some(_) | None => {
                        err.span_note(span, &unknown_scope());
                        return;
                    }
                };
                let scope_decorated_tag = match scope.data() {
                    region::ScopeData::Node(_) => tag,
                    region::ScopeData::CallSite(_) => "scope of call-site for function",
                    region::ScopeData::Arguments(_) => "scope of function body",
                    region::ScopeData::Destruction(_) => {
                        new_string = format!("destruction scope surrounding {}", tag);
                        &new_string[..]
                    }
                    region::ScopeData::Remainder(r) => {
                        new_string = format!(
                            "block suffix following statement {}",
                            r.first_statement_index.index()
                        );
                        &new_string[..]
                    }
                };
                self.explain_span(scope_decorated_tag, span)
            }

            ty::ReEarlyBound(_) | ty::ReFree(_) | ty::ReStatic => {
                self.msg_span_from_free_region(region)
            }

            ty::ReEmpty => ("the empty lifetime".to_owned(), None),

            // FIXME(#13998) ReSkolemized should probably print like
            // ReFree rather than dumping Debug output on the user.
            //
            // We shouldn't really be having unification failures with ReVar
            // and ReLateBound though.
            ty::ReSkolemized(..) | ty::ReVar(_) | ty::ReLateBound(..) | ty::ReErased => {
                (format!("lifetime {:?}", region), None)
            }

            // We shouldn't encounter an error message with ReClosureBound.
            ty::ReCanonical(..) |
            ty::ReClosureBound(..) => {
                bug!("encountered unexpected ReClosureBound: {:?}", region,);
            }
        };

        TyCtxt::emit_msg_span(err, prefix, description, span, suffix);
    }

    pub fn note_and_explain_free_region(
        self,
        err: &mut DiagnosticBuilder,
        prefix: &str,
        region: ty::Region<'tcx>,
        suffix: &str,
    ) {
        let (description, span) = self.msg_span_from_free_region(region);

        TyCtxt::emit_msg_span(err, prefix, description, span, suffix);
    }

    fn msg_span_from_free_region(self, region: ty::Region<'tcx>) -> (String, Option<Span>) {
        match *region {
            ty::ReEarlyBound(_) | ty::ReFree(_)  => {
                self.msg_span_from_early_bound_and_free_regions(region)
            },
            ty::ReStatic => ("the static lifetime".to_owned(), None),
            _ => bug!("{:?}", region),
        }
    }

    fn msg_span_from_early_bound_and_free_regions(
        self,
        region: ty::Region<'tcx>,
    ) -> (String, Option<Span>) {
        let scope = region.free_region_binding_scope(self);
        let node = self.hir.as_local_node_id(scope).unwrap_or(DUMMY_NODE_ID);
        let unknown;
        let tag = match self.hir.find(node) {
            Some(hir_map::NodeBlock(_)) | Some(hir_map::NodeExpr(_)) => "body",
            Some(hir_map::NodeItem(it)) => Self::item_scope_tag(&it),
            Some(hir_map::NodeTraitItem(it)) => Self::trait_item_scope_tag(&it),
            Some(hir_map::NodeImplItem(it)) => Self::impl_item_scope_tag(&it),

            // this really should not happen, but it does:
            // FIXME(#27942)
            Some(_) => {
                unknown = format!(
                    "unexpected node ({}) for scope {:?}.  \
                     Please report a bug.",
                    self.hir.node_to_string(node),
                    scope
                );
                &unknown
            }
            None => {
                unknown = format!(
                    "unknown node for scope {:?}.  \
                     Please report a bug.",
                    scope
                );
                &unknown
            }
        };
        let (prefix, span) = match *region {
            ty::ReEarlyBound(ref br) => (
                format!("the lifetime {} as defined on", br.name),
                self.sess.codemap().def_span(self.hir.span(node)),
            ),
            ty::ReFree(ref fr) => match fr.bound_region {
                ty::BrAnon(idx) => (
                    format!("the anonymous lifetime #{} defined on", idx + 1),
                    self.hir.span(node),
                ),
                ty::BrFresh(_) => (
                    "an anonymous lifetime defined on".to_owned(),
                    self.hir.span(node),
                ),
                _ => (
                    format!("the lifetime {} as defined on", fr.bound_region),
                    self.sess.codemap().def_span(self.hir.span(node)),
                ),
            },
            _ => bug!(),
        };
        let (msg, opt_span) = self.explain_span(tag, span);
        (format!("{} {}", prefix, msg), opt_span)
    }

    fn emit_msg_span(
        err: &mut DiagnosticBuilder,
        prefix: &str,
        description: String,
        span: Option<Span>,
        suffix: &str,
    ) {
        let message = format!("{}{}{}", prefix, description, suffix);

        if let Some(span) = span {
            err.span_note(span, &message);
        } else {
            err.note(&message);
        }
    }

    fn item_scope_tag(item: &hir::Item) -> &'static str {
        match item.node {
            hir::ItemImpl(..) => "impl",
            hir::ItemStruct(..) => "struct",
            hir::ItemUnion(..) => "union",
            hir::ItemEnum(..) => "enum",
            hir::ItemTrait(..) => "trait",
            hir::ItemFn(..) => "function body",
            _ => "item",
        }
    }

    fn trait_item_scope_tag(item: &hir::TraitItem) -> &'static str {
        match item.node {
            hir::TraitItemKind::Method(..) => "method body",
            hir::TraitItemKind::Const(..) | hir::TraitItemKind::Type(..) => "associated item",
        }
    }

    fn impl_item_scope_tag(item: &hir::ImplItem) -> &'static str {
        match item.node {
            hir::ImplItemKind::Method(..) => "method body",
            hir::ImplItemKind::Const(..) | hir::ImplItemKind::Type(_) => "associated item",
        }
    }

    fn explain_span(self, heading: &str, span: Span) -> (String, Option<Span>) {
        let lo = self.sess.codemap().lookup_char_pos_adj(span.lo());
        (
            format!("the {} at {}:{}", heading, lo.line, lo.col.to_usize() + 1),
            Some(span),
        )
    }
}

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    pub fn report_region_errors(
        &self,
        region_scope_tree: &region::ScopeTree,
        errors: &Vec<RegionResolutionError<'tcx>>,
        will_later_be_reported_by_nll: bool,
    ) {
        debug!("report_region_errors(): {} errors to start", errors.len());

        if will_later_be_reported_by_nll && self.tcx.use_mir_borrowck() {
            // With `#![feature(nll)]`, we want to present a nice user
            // experience, so don't even mention the errors from the
            // AST checker.
            if self.tcx.features().nll {
                return;
            }

            // But with nll, it's nice to have some note for later.
            for error in errors {
                match *error {
                    RegionResolutionError::ConcreteFailure(ref origin, ..)
                    | RegionResolutionError::GenericBoundFailure(ref origin, ..) => {
                        self.tcx
                            .sess
                            .span_warn(origin.span(), "not reporting region error due to nll");
                    }

                    RegionResolutionError::SubSupConflict(ref rvo, ..) => {
                        self.tcx
                            .sess
                            .span_warn(rvo.span(), "not reporting region error due to nll");
                    }
                }
            }

            return;
        }

        // try to pre-process the errors, which will group some of them
        // together into a `ProcessedErrors` group:
        let errors = self.process_errors(errors);

        debug!(
            "report_region_errors: {} errors after preprocessing",
            errors.len()
        );

        for error in errors {
            debug!("report_region_errors: error = {:?}", error);

            if !self.try_report_nice_region_error(&error) {
                match error.clone() {
                    // These errors could indicate all manner of different
                    // problems with many different solutions. Rather
                    // than generate a "one size fits all" error, what we
                    // attempt to do is go through a number of specific
                    // scenarios and try to find the best way to present
                    // the error. If all of these fails, we fall back to a rather
                    // general bit of code that displays the error information
                    RegionResolutionError::ConcreteFailure(origin, sub, sup) => {
                        self.report_concrete_failure(region_scope_tree, origin, sub, sup)
                            .emit();
                    }

                    RegionResolutionError::GenericBoundFailure(origin, param_ty, sub) => {
                        self.report_generic_bound_failure(
                            region_scope_tree,
                            origin.span(),
                            Some(origin),
                            param_ty,
                            sub,
                        );
                    }

                    RegionResolutionError::SubSupConflict(
                        var_origin,
                        sub_origin,
                        sub_r,
                        sup_origin,
                        sup_r,
                    ) => {
                        self.report_sub_sup_conflict(
                            region_scope_tree,
                            var_origin,
                            sub_origin,
                            sub_r,
                            sup_origin,
                            sup_r,
                        );
                    }
                }
            }
        }
    }

    // This method goes through all the errors and try to group certain types
    // of error together, for the purpose of suggesting explicit lifetime
    // parameters to the user. This is done so that we can have a more
    // complete view of what lifetimes should be the same.
    // If the return value is an empty vector, it means that processing
    // failed (so the return value of this method should not be used).
    //
    // The method also attempts to weed out messages that seem like
    // duplicates that will be unhelpful to the end-user. But
    // obviously it never weeds out ALL errors.
    fn process_errors(
        &self,
        errors: &Vec<RegionResolutionError<'tcx>>,
    ) -> Vec<RegionResolutionError<'tcx>> {
        debug!("process_errors()");

        // We want to avoid reporting generic-bound failures if we can
        // avoid it: these have a very high rate of being unhelpful in
        // practice. This is because they are basically secondary
        // checks that test the state of the region graph after the
        // rest of inference is done, and the other kinds of errors
        // indicate that the region constraint graph is internally
        // inconsistent, so these test results are likely to be
        // meaningless.
        //
        // Therefore, we filter them out of the list unless they are
        // the only thing in the list.

        let is_bound_failure = |e: &RegionResolutionError<'tcx>| match *e {
            RegionResolutionError::GenericBoundFailure(..) => true,
            RegionResolutionError::ConcreteFailure(..)
            | RegionResolutionError::SubSupConflict(..) => false,
        };

        let mut errors = if errors.iter().all(|e| is_bound_failure(e)) {
            errors.clone()
        } else {
            errors
                .iter()
                .filter(|&e| !is_bound_failure(e))
                .cloned()
                .collect()
        };

        // sort the errors by span, for better error message stability.
        errors.sort_by_key(|u| match *u {
            RegionResolutionError::ConcreteFailure(ref sro, _, _) => sro.span(),
            RegionResolutionError::GenericBoundFailure(ref sro, _, _) => sro.span(),
            RegionResolutionError::SubSupConflict(ref rvo, _, _, _, _) => rvo.span(),
        });
        errors
    }

    /// Adds a note if the types come from similarly named crates
    fn check_and_note_conflicting_crates(
        &self,
        err: &mut DiagnosticBuilder,
        terr: &TypeError<'tcx>,
        sp: Span,
    ) {
        let report_path_match = |err: &mut DiagnosticBuilder, did1: DefId, did2: DefId| {
            // Only external crates, if either is from a local
            // module we could have false positives
            if !(did1.is_local() || did2.is_local()) && did1.krate != did2.krate {
                let exp_path = self.tcx.item_path_str(did1);
                let found_path = self.tcx.item_path_str(did2);
                let exp_abs_path = self.tcx.absolute_item_path_str(did1);
                let found_abs_path = self.tcx.absolute_item_path_str(did2);
                // We compare strings because DefPath can be different
                // for imported and non-imported crates
                if exp_path == found_path || exp_abs_path == found_abs_path {
                    let crate_name = self.tcx.crate_name(did1.krate);
                    err.span_note(
                        sp,
                        &format!(
                            "Perhaps two different versions \
                             of crate `{}` are being used?",
                            crate_name
                        ),
                    );
                }
            }
        };
        match *terr {
            TypeError::Sorts(ref exp_found) => {
                // if they are both "path types", there's a chance of ambiguity
                // due to different versions of the same crate
                match (&exp_found.expected.sty, &exp_found.found.sty) {
                    (&ty::TyAdt(exp_adt, _), &ty::TyAdt(found_adt, _)) => {
                        report_path_match(err, exp_adt.did, found_adt.did);
                    }
                    _ => (),
                }
            }
            TypeError::Traits(ref exp_found) => {
                report_path_match(err, exp_found.expected, exp_found.found);
            }
            _ => (), // FIXME(#22750) handle traits and stuff
        }
    }

    fn note_error_origin(&self, err: &mut DiagnosticBuilder<'tcx>, cause: &ObligationCause<'tcx>) {
        match cause.code {
            ObligationCauseCode::MatchExpressionArm { arm_span, source } => match source {
                hir::MatchSource::IfLetDesugar { .. } => {
                    let msg = "`if let` arm with an incompatible type";
                    if self.tcx.sess.codemap().is_multiline(arm_span) {
                        err.span_note(arm_span, msg);
                    } else {
                        err.span_label(arm_span, msg);
                    }
                }
                _ => {
                    let msg = "match arm with an incompatible type";
                    if self.tcx.sess.codemap().is_multiline(arm_span) {
                        err.span_note(arm_span, msg);
                    } else {
                        err.span_label(arm_span, msg);
                    }
                }
            },
            _ => (),
        }
    }

    /// Given that `other_ty` is the same as a type argument for `name` in `sub`, populate `value`
    /// highlighting `name` and every type argument that isn't at `pos` (which is `other_ty`), and
    /// populate `other_value` with `other_ty`.
    ///
    /// ```text
    /// Foo<Bar<Qux>>
    /// ^^^^--------^ this is highlighted
    /// |   |
    /// |   this type argument is exactly the same as the other type, not highlighted
    /// this is highlighted
    /// Bar<Qux>
    /// -------- this type is the same as a type argument in the other type, not highlighted
    /// ```
    fn highlight_outer(
        &self,
        value: &mut DiagnosticStyledString,
        other_value: &mut DiagnosticStyledString,
        name: String,
        sub: &ty::subst::Substs<'tcx>,
        pos: usize,
        other_ty: &Ty<'tcx>,
    ) {
        // `value` and `other_value` hold two incomplete type representation for display.
        // `name` is the path of both types being compared. `sub`
        value.push_highlighted(name);
        let len = sub.len();
        if len > 0 {
            value.push_highlighted("<");
        }

        // Output the lifetimes fot the first type
        let lifetimes = sub.regions()
            .map(|lifetime| {
                let s = format!("{}", lifetime);
                if s.is_empty() {
                    "'_".to_string()
                } else {
                    s
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        if !lifetimes.is_empty() {
            if sub.regions().count() < len {
                value.push_normal(lifetimes + &", ");
            } else {
                value.push_normal(lifetimes);
            }
        }

        // Highlight all the type arguments that aren't at `pos` and compare the type argument at
        // `pos` and `other_ty`.
        for (i, type_arg) in sub.types().enumerate() {
            if i == pos {
                let values = self.cmp(type_arg, other_ty);
                value.0.extend((values.0).0);
                other_value.0.extend((values.1).0);
            } else {
                value.push_highlighted(format!("{}", type_arg));
            }

            if len > 0 && i != len - 1 {
                value.push_normal(", ");
            }
            //self.push_comma(&mut value, &mut other_value, len, i);
        }
        if len > 0 {
            value.push_highlighted(">");
        }
    }

    /// If `other_ty` is the same as a type argument present in `sub`, highlight `path` in `t1_out`,
    /// as that is the difference to the other type.
    ///
    /// For the following code:
    ///
    /// ```norun
    /// let x: Foo<Bar<Qux>> = foo::<Bar<Qux>>();
    /// ```
    ///
    /// The type error output will behave in the following way:
    ///
    /// ```text
    /// Foo<Bar<Qux>>
    /// ^^^^--------^ this is highlighted
    /// |   |
    /// |   this type argument is exactly the same as the other type, not highlighted
    /// this is highlighted
    /// Bar<Qux>
    /// -------- this type is the same as a type argument in the other type, not highlighted
    /// ```
    fn cmp_type_arg(
        &self,
        mut t1_out: &mut DiagnosticStyledString,
        mut t2_out: &mut DiagnosticStyledString,
        path: String,
        sub: &ty::subst::Substs<'tcx>,
        other_path: String,
        other_ty: &Ty<'tcx>,
    ) -> Option<()> {
        for (i, ta) in sub.types().enumerate() {
            if &ta == other_ty {
                self.highlight_outer(&mut t1_out, &mut t2_out, path, sub, i, &other_ty);
                return Some(());
            }
            if let &ty::TyAdt(def, _) = &ta.sty {
                let path_ = self.tcx.item_path_str(def.did.clone());
                if path_ == other_path {
                    self.highlight_outer(&mut t1_out, &mut t2_out, path, sub, i, &other_ty);
                    return Some(());
                }
            }
        }
        None
    }

    /// Add a `,` to the type representation only if it is appropriate.
    fn push_comma(
        &self,
        value: &mut DiagnosticStyledString,
        other_value: &mut DiagnosticStyledString,
        len: usize,
        pos: usize,
    ) {
        if len > 0 && pos != len - 1 {
            value.push_normal(", ");
            other_value.push_normal(", ");
        }
    }

    /// Compare two given types, eliding parts that are the same between them and highlighting
    /// relevant differences, and return two representation of those types for highlighted printing.
    fn cmp(&self, t1: Ty<'tcx>, t2: Ty<'tcx>) -> (DiagnosticStyledString, DiagnosticStyledString) {
        fn equals<'tcx>(a: &Ty<'tcx>, b: &Ty<'tcx>) -> bool {
            match (&a.sty, &b.sty) {
                (a, b) if *a == *b => true,
                (&ty::TyInt(_), &ty::TyInfer(ty::InferTy::IntVar(_)))
                | (&ty::TyInfer(ty::InferTy::IntVar(_)), &ty::TyInt(_))
                | (&ty::TyInfer(ty::InferTy::IntVar(_)), &ty::TyInfer(ty::InferTy::IntVar(_)))
                | (&ty::TyFloat(_), &ty::TyInfer(ty::InferTy::FloatVar(_)))
                | (&ty::TyInfer(ty::InferTy::FloatVar(_)), &ty::TyFloat(_))
                | (
                    &ty::TyInfer(ty::InferTy::FloatVar(_)),
                    &ty::TyInfer(ty::InferTy::FloatVar(_)),
                ) => true,
                _ => false,
            }
        }

        fn push_ty_ref<'tcx>(
            r: &ty::Region<'tcx>,
            ty: Ty<'tcx>,
            mutbl: hir::Mutability,
            s: &mut DiagnosticStyledString,
        ) {
            let r = &format!("{}", r);
            s.push_highlighted(format!(
                "&{}{}{}",
                r,
                if r == "" { "" } else { " " },
                if mutbl == hir::MutMutable {
                    "mut "
                } else {
                    ""
                }
            ));
            s.push_normal(format!("{}", ty));
        }

        match (&t1.sty, &t2.sty) {
            (&ty::TyAdt(def1, sub1), &ty::TyAdt(def2, sub2)) => {
                let mut values = (DiagnosticStyledString::new(), DiagnosticStyledString::new());
                let path1 = self.tcx.item_path_str(def1.did.clone());
                let path2 = self.tcx.item_path_str(def2.did.clone());
                if def1.did == def2.did {
                    // Easy case. Replace same types with `_` to shorten the output and highlight
                    // the differing ones.
                    //     let x: Foo<Bar, Qux> = y::<Foo<Quz, Qux>>();
                    //     Foo<Bar, _>
                    //     Foo<Quz, _>
                    //         ---  ^ type argument elided
                    //         |
                    //         highlighted in output
                    values.0.push_normal(path1);
                    values.1.push_normal(path2);

                    // Only draw `<...>` if there're lifetime/type arguments.
                    let len = sub1.len();
                    if len > 0 {
                        values.0.push_normal("<");
                        values.1.push_normal("<");
                    }

                    fn lifetime_display(lifetime: Region) -> String {
                        let s = format!("{}", lifetime);
                        if s.is_empty() {
                            "'_".to_string()
                        } else {
                            s
                        }
                    }
                    // At one point we'd like to elide all lifetimes here, they are irrelevant for
                    // all diagnostics that use this output
                    //
                    //     Foo<'x, '_, Bar>
                    //     Foo<'y, '_, Qux>
                    //         ^^  ^^  --- type arguments are not elided
                    //         |   |
                    //         |   elided as they were the same
                    //         not elided, they were different, but irrelevant
                    let lifetimes = sub1.regions().zip(sub2.regions());
                    for (i, lifetimes) in lifetimes.enumerate() {
                        let l1 = lifetime_display(lifetimes.0);
                        let l2 = lifetime_display(lifetimes.1);
                        if l1 == l2 {
                            values.0.push_normal("'_");
                            values.1.push_normal("'_");
                        } else {
                            values.0.push_highlighted(l1);
                            values.1.push_highlighted(l2);
                        }
                        self.push_comma(&mut values.0, &mut values.1, len, i);
                    }

                    // We're comparing two types with the same path, so we compare the type
                    // arguments for both. If they are the same, do not highlight and elide from the
                    // output.
                    //     Foo<_, Bar>
                    //     Foo<_, Qux>
                    //         ^ elided type as this type argument was the same in both sides
                    let type_arguments = sub1.types().zip(sub2.types());
                    let regions_len = sub1.regions().collect::<Vec<_>>().len();
                    for (i, (ta1, ta2)) in type_arguments.enumerate() {
                        let i = i + regions_len;
                        if ta1 == ta2 {
                            values.0.push_normal("_");
                            values.1.push_normal("_");
                        } else {
                            let (x1, x2) = self.cmp(ta1, ta2);
                            (values.0).0.extend(x1.0);
                            (values.1).0.extend(x2.0);
                        }
                        self.push_comma(&mut values.0, &mut values.1, len, i);
                    }

                    // Close the type argument bracket.
                    // Only draw `<...>` if there're lifetime/type arguments.
                    if len > 0 {
                        values.0.push_normal(">");
                        values.1.push_normal(">");
                    }
                    values
                } else {
                    // Check for case:
                    //     let x: Foo<Bar<Qux> = foo::<Bar<Qux>>();
                    //     Foo<Bar<Qux>
                    //         ------- this type argument is exactly the same as the other type
                    //     Bar<Qux>
                    if self.cmp_type_arg(
                        &mut values.0,
                        &mut values.1,
                        path1.clone(),
                        sub1,
                        path2.clone(),
                        &t2,
                    ).is_some()
                    {
                        return values;
                    }
                    // Check for case:
                    //     let x: Bar<Qux> = y:<Foo<Bar<Qux>>>();
                    //     Bar<Qux>
                    //     Foo<Bar<Qux>>
                    //         ------- this type argument is exactly the same as the other type
                    if self.cmp_type_arg(&mut values.1, &mut values.0, path2, sub2, path1, &t1)
                        .is_some()
                    {
                        return values;
                    }

                    // We couldn't find anything in common, highlight everything.
                    //     let x: Bar<Qux> = y::<Foo<Zar>>();
                    (
                        DiagnosticStyledString::highlighted(format!("{}", t1)),
                        DiagnosticStyledString::highlighted(format!("{}", t2)),
                    )
                }
            }

            // When finding T != &T, highlight only the borrow
            (&ty::TyRef(r1, ref_ty1, mutbl1), _) if equals(&ref_ty1, &t2) => {
                let mut values = (DiagnosticStyledString::new(), DiagnosticStyledString::new());
                push_ty_ref(&r1, ref_ty1, mutbl1, &mut values.0);
                values.1.push_normal(format!("{}", t2));
                values
            }
            (_, &ty::TyRef(r2, ref_ty2, mutbl2)) if equals(&t1, &ref_ty2) => {
                let mut values = (DiagnosticStyledString::new(), DiagnosticStyledString::new());
                values.0.push_normal(format!("{}", t1));
                push_ty_ref(&r2, ref_ty2, mutbl2, &mut values.1);
                values
            }

            // When encountering &T != &mut T, highlight only the borrow
            (&ty::TyRef(r1, ref_ty1, mutbl1),
             &ty::TyRef(r2, ref_ty2, mutbl2)) if equals(&ref_ty1, &ref_ty2) => {
                let mut values = (DiagnosticStyledString::new(), DiagnosticStyledString::new());
                push_ty_ref(&r1, ref_ty1, mutbl1, &mut values.0);
                push_ty_ref(&r2, ref_ty2, mutbl2, &mut values.1);
                values
            }

            _ => {
                if t1 == t2 {
                    // The two types are the same, elide and don't highlight.
                    (
                        DiagnosticStyledString::normal("_"),
                        DiagnosticStyledString::normal("_"),
                    )
                } else {
                    // We couldn't find anything in common, highlight everything.
                    (
                        DiagnosticStyledString::highlighted(format!("{}", t1)),
                        DiagnosticStyledString::highlighted(format!("{}", t2)),
                    )
                }
            }
        }
    }

    pub fn note_type_err(
        &self,
        diag: &mut DiagnosticBuilder<'tcx>,
        cause: &ObligationCause<'tcx>,
        secondary_span: Option<(Span, String)>,
        mut values: Option<ValuePairs<'tcx>>,
        terr: &TypeError<'tcx>,
    ) {
        // For some types of errors, expected-found does not make
        // sense, so just ignore the values we were given.
        match terr {
            TypeError::CyclicTy(_) => {
                values = None;
            }
            _ => {}
        }

        let (expected_found, exp_found, is_simple_error) = match values {
            None => (None, None, false),
            Some(values) => {
                let (is_simple_error, exp_found) = match values {
                    ValuePairs::Types(exp_found) => {
                        let is_simple_err =
                            exp_found.expected.is_primitive() && exp_found.found.is_primitive();

                        (is_simple_err, Some(exp_found))
                    }
                    _ => (false, None),
                };
                let vals = match self.values_str(&values) {
                    Some((expected, found)) => Some((expected, found)),
                    None => {
                        // Derived error. Cancel the emitter.
                        self.tcx.sess.diagnostic().cancel(diag);
                        return;
                    }
                };
                (vals, exp_found, is_simple_error)
            }
        };

        let span = cause.span(&self.tcx);

        diag.span_label(span, terr.to_string());
        if let Some((sp, msg)) = secondary_span {
            diag.span_label(sp, msg);
        }

        if let Some((expected, found)) = expected_found {
            match (terr, is_simple_error, expected == found) {
                (&TypeError::Sorts(ref values), false, true) => {
                    diag.note_expected_found_extra(
                        &"type",
                        expected,
                        found,
                        &format!(" ({})", values.expected.sort_string(self.tcx)),
                        &format!(" ({})", values.found.sort_string(self.tcx)),
                    );
                }
                (_, false, _) => {
                    if let Some(exp_found) = exp_found {
                        let (def_id, ret_ty) = match exp_found.found.sty {
                            TypeVariants::TyFnDef(def, _) => {
                                (Some(def), Some(self.tcx.fn_sig(def).output()))
                            }
                            _ => (None, None),
                        };

                        let exp_is_struct = match exp_found.expected.sty {
                            TypeVariants::TyAdt(def, _) => def.is_struct(),
                            _ => false,
                        };

                        if let (Some(def_id), Some(ret_ty)) = (def_id, ret_ty) {
                            if exp_is_struct && &exp_found.expected == ret_ty.skip_binder() {
                                let message = format!(
                                    "did you mean `{}(/* fields */)`?",
                                    self.tcx.item_path_str(def_id)
                                );
                                diag.span_label(span, message);
                            }
                        }
                    }

                    diag.note_expected_found(&"type", expected, found);
                }
                _ => (),
            }
        }

        self.check_and_note_conflicting_crates(diag, terr, span);
        self.tcx.note_and_explain_type_err(diag, terr, span);

        // It reads better to have the error origin as the final
        // thing.
        self.note_error_origin(diag, &cause);
    }

    pub fn report_and_explain_type_error(
        &self,
        trace: TypeTrace<'tcx>,
        terr: &TypeError<'tcx>,
    ) -> DiagnosticBuilder<'tcx> {
        debug!(
            "report_and_explain_type_error(trace={:?}, terr={:?})",
            trace, terr
        );

        let span = trace.cause.span(&self.tcx);
        let failure_code = trace.cause.as_failure_code(terr);
        let mut diag = match failure_code {
            FailureCode::Error0317(failure_str) => {
                struct_span_err!(self.tcx.sess, span, E0317, "{}", failure_str)
            }
            FailureCode::Error0580(failure_str) => {
                struct_span_err!(self.tcx.sess, span, E0580, "{}", failure_str)
            }
            FailureCode::Error0308(failure_str) => {
                struct_span_err!(self.tcx.sess, span, E0308, "{}", failure_str)
            }
            FailureCode::Error0644(failure_str) => {
                struct_span_err!(self.tcx.sess, span, E0644, "{}", failure_str)
            }
        };
        self.note_type_err(&mut diag, &trace.cause, None, Some(trace.values), terr);
        diag
    }

    fn values_str(
        &self,
        values: &ValuePairs<'tcx>,
    ) -> Option<(DiagnosticStyledString, DiagnosticStyledString)> {
        match *values {
            infer::Types(ref exp_found) => self.expected_found_str_ty(exp_found),
            infer::Regions(ref exp_found) => self.expected_found_str(exp_found),
            infer::TraitRefs(ref exp_found) => self.expected_found_str(exp_found),
            infer::PolyTraitRefs(ref exp_found) => self.expected_found_str(exp_found),
        }
    }

    fn expected_found_str_ty(
        &self,
        exp_found: &ty::error::ExpectedFound<Ty<'tcx>>,
    ) -> Option<(DiagnosticStyledString, DiagnosticStyledString)> {
        let exp_found = self.resolve_type_vars_if_possible(exp_found);
        if exp_found.references_error() {
            return None;
        }

        Some(self.cmp(exp_found.expected, exp_found.found))
    }

    /// Returns a string of the form "expected `{}`, found `{}`".
    fn expected_found_str<T: fmt::Display + TypeFoldable<'tcx>>(
        &self,
        exp_found: &ty::error::ExpectedFound<T>,
    ) -> Option<(DiagnosticStyledString, DiagnosticStyledString)> {
        let exp_found = self.resolve_type_vars_if_possible(exp_found);
        if exp_found.references_error() {
            return None;
        }

        Some((
            DiagnosticStyledString::highlighted(format!("{}", exp_found.expected)),
            DiagnosticStyledString::highlighted(format!("{}", exp_found.found)),
        ))
    }

    pub fn report_generic_bound_failure(
        &self,
        region_scope_tree: &region::ScopeTree,
        span: Span,
        origin: Option<SubregionOrigin<'tcx>>,
        bound_kind: GenericKind<'tcx>,
        sub: Region<'tcx>,
    ) {
        // Attempt to obtain the span of the parameter so we can
        // suggest adding an explicit lifetime bound to it.
        let type_param_span = match (self.in_progress_tables, bound_kind) {
            (Some(ref table), GenericKind::Param(ref param)) => {
                let table = table.borrow();
                table.local_id_root.and_then(|did| {
                    let generics = self.tcx.generics_of(did);
                    // Account for the case where `did` corresponds to `Self`, which doesn't have
                    // the expected type argument.
                    if !param.is_self() {
                        let type_param = generics.type_param(param, self.tcx);
                        let hir = &self.tcx.hir;
                        hir.as_local_node_id(type_param.def_id).map(|id| {
                            // Get the `hir::TyParam` to verify whether it already has any bounds.
                            // We do this to avoid suggesting code that ends up as `T: 'a'b`,
                            // instead we suggest `T: 'a + 'b` in that case.
                            let mut has_bounds = false;
                            if let hir_map::NodeGenericParam(ref param) = hir.get(id) {
                                has_bounds = !param.bounds.is_empty();
                            }
                            let sp = hir.span(id);
                            // `sp` only covers `T`, change it so that it covers
                            // `T:` when appropriate
                            let sp = if has_bounds {
                                sp.to(self.tcx
                                    .sess
                                    .codemap()
                                    .next_point(self.tcx.sess.codemap().next_point(sp)))
                            } else {
                                sp
                            };
                            (sp, has_bounds)
                        })
                    } else {
                        None
                    }
                })
            }
            _ => None,
        };

        let labeled_user_string = match bound_kind {
            GenericKind::Param(ref p) => format!("the parameter type `{}`", p),
            GenericKind::Projection(ref p) => format!("the associated type `{}`", p),
        };

        if let Some(SubregionOrigin::CompareImplMethodObligation {
            span,
            item_name,
            impl_item_def_id,
            trait_item_def_id,
        }) = origin
        {
            self.report_extra_impl_obligation(
                span,
                item_name,
                impl_item_def_id,
                trait_item_def_id,
                &format!("`{}: {}`", bound_kind, sub),
            ).emit();
            return;
        }

        fn binding_suggestion<'tcx, S: fmt::Display>(
            err: &mut DiagnosticBuilder<'tcx>,
            type_param_span: Option<(Span, bool)>,
            bound_kind: GenericKind<'tcx>,
            sub: S,
        ) {
            let consider = &format!(
                "consider adding an explicit lifetime bound `{}: {}`...",
                bound_kind, sub
            );
            if let Some((sp, has_lifetimes)) = type_param_span {
                let tail = if has_lifetimes { " + " } else { "" };
                let suggestion = format!("{}: {}{}", bound_kind, sub, tail);
                err.span_suggestion_short_with_applicability(
                    sp, consider, suggestion,
                    Applicability::MaybeIncorrect // Issue #41966
                );
            } else {
                err.help(consider);
            }
        }

        let mut err = match *sub {
            ty::ReEarlyBound(_)
            | ty::ReFree(ty::FreeRegion {
                bound_region: ty::BrNamed(..),
                ..
            }) => {
                // Does the required lifetime have a nice name we can print?
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0309,
                    "{} may not live long enough",
                    labeled_user_string
                );
                binding_suggestion(&mut err, type_param_span, bound_kind, sub);
                err
            }

            ty::ReStatic => {
                // Does the required lifetime have a nice name we can print?
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0310,
                    "{} may not live long enough",
                    labeled_user_string
                );
                binding_suggestion(&mut err, type_param_span, bound_kind, "'static");
                err
            }

            _ => {
                // If not, be less specific.
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0311,
                    "{} may not live long enough",
                    labeled_user_string
                );
                err.help(&format!(
                    "consider adding an explicit lifetime bound for `{}`",
                    bound_kind
                ));
                self.tcx.note_and_explain_region(
                    region_scope_tree,
                    &mut err,
                    &format!("{} must be valid for ", labeled_user_string),
                    sub,
                    "...",
                );
                err
            }
        };

        if let Some(origin) = origin {
            self.note_region_origin(&mut err, &origin);
        }
        err.emit();
    }

    fn report_sub_sup_conflict(
        &self,
        region_scope_tree: &region::ScopeTree,
        var_origin: RegionVariableOrigin,
        sub_origin: SubregionOrigin<'tcx>,
        sub_region: Region<'tcx>,
        sup_origin: SubregionOrigin<'tcx>,
        sup_region: Region<'tcx>,
    ) {
        let mut err = self.report_inference_failure(var_origin);

        self.tcx.note_and_explain_region(
            region_scope_tree,
            &mut err,
            "first, the lifetime cannot outlive ",
            sup_region,
            "...",
        );

        match (&sup_origin, &sub_origin) {
            (&infer::Subtype(ref sup_trace), &infer::Subtype(ref sub_trace)) => {
                if let (Some((sup_expected, sup_found)), Some((sub_expected, sub_found))) = (
                    self.values_str(&sup_trace.values),
                    self.values_str(&sub_trace.values),
                ) {
                    if sub_expected == sup_expected && sub_found == sup_found {
                        self.tcx.note_and_explain_region(
                            region_scope_tree,
                            &mut err,
                            "...but the lifetime must also be valid for ",
                            sub_region,
                            "...",
                        );
                        err.note(&format!(
                            "...so that the {}:\nexpected {}\n   found {}",
                            sup_trace.cause.as_requirement_str(),
                            sup_expected.content(),
                            sup_found.content()
                        ));
                        err.emit();
                        return;
                    }
                }
            }
            _ => {}
        }

        self.note_region_origin(&mut err, &sup_origin);

        self.tcx.note_and_explain_region(
            region_scope_tree,
            &mut err,
            "but, the lifetime must be valid for ",
            sub_region,
            "...",
        );

        self.note_region_origin(&mut err, &sub_origin);
        err.emit();
    }
}

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    fn report_inference_failure(
        &self,
        var_origin: RegionVariableOrigin,
    ) -> DiagnosticBuilder<'tcx> {
        let br_string = |br: ty::BoundRegion| {
            let mut s = br.to_string();
            if !s.is_empty() {
                s.push_str(" ");
            }
            s
        };
        let var_description = match var_origin {
            infer::MiscVariable(_) => "".to_string(),
            infer::PatternRegion(_) => " for pattern".to_string(),
            infer::AddrOfRegion(_) => " for borrow expression".to_string(),
            infer::Autoref(_) => " for autoref".to_string(),
            infer::Coercion(_) => " for automatic coercion".to_string(),
            infer::LateBoundRegion(_, br, infer::FnCall) => {
                format!(" for lifetime parameter {}in function call", br_string(br))
            }
            infer::LateBoundRegion(_, br, infer::HigherRankedType) => {
                format!(" for lifetime parameter {}in generic type", br_string(br))
            }
            infer::LateBoundRegion(_, br, infer::AssocTypeProjection(def_id)) => format!(
                " for lifetime parameter {}in trait containing associated type `{}`",
                br_string(br),
                self.tcx.associated_item(def_id).name
            ),
            infer::EarlyBoundRegion(_, name) => format!(" for lifetime parameter `{}`", name),
            infer::BoundRegionInCoherence(name) => {
                format!(" for lifetime parameter `{}` in coherence check", name)
            }
            infer::UpvarRegion(ref upvar_id, _) => {
                let var_node_id = self.tcx.hir.hir_to_node_id(upvar_id.var_id);
                let var_name = self.tcx.hir.name(var_node_id);
                format!(" for capture of `{}` by closure", var_name)
            }
            infer::NLL(..) => bug!("NLL variable found in lexical phase"),
        };

        struct_span_err!(
            self.tcx.sess,
            var_origin.span(),
            E0495,
            "cannot infer an appropriate lifetime{} \
             due to conflicting requirements",
            var_description
        )
    }
}

enum FailureCode {
    Error0317(&'static str),
    Error0580(&'static str),
    Error0308(&'static str),
    Error0644(&'static str),
}

impl<'tcx> ObligationCause<'tcx> {
    fn as_failure_code(&self, terr: &TypeError<'tcx>) -> FailureCode {
        use self::FailureCode::*;
        use traits::ObligationCauseCode::*;
        match self.code {
            CompareImplMethodObligation { .. } => Error0308("method not compatible with trait"),
            MatchExpressionArm { source, .. } => Error0308(match source {
                hir::MatchSource::IfLetDesugar { .. } => "`if let` arms have incompatible types",
                _ => "match arms have incompatible types",
            }),
            IfExpression => Error0308("if and else have incompatible types"),
            IfExpressionWithNoElse => Error0317("if may be missing an else clause"),
            MainFunctionType => Error0580("main function has wrong type"),
            StartFunctionType => Error0308("start function has wrong type"),
            IntrinsicType => Error0308("intrinsic has wrong type"),
            MethodReceiver => Error0308("mismatched method receiver"),

            // In the case where we have no more specific thing to
            // say, also take a look at the error code, maybe we can
            // tailor to that.
            _ => match terr {
                TypeError::CyclicTy(ty) if ty.is_closure() || ty.is_generator() => {
                    Error0644("closure/generator type that references itself")
                }
                _ => Error0308("mismatched types"),
            },
        }
    }

    fn as_requirement_str(&self) -> &'static str {
        use traits::ObligationCauseCode::*;
        match self.code {
            CompareImplMethodObligation { .. } => "method type is compatible with trait",
            ExprAssignable => "expression is assignable",
            MatchExpressionArm { source, .. } => match source {
                hir::MatchSource::IfLetDesugar { .. } => "`if let` arms have compatible types",
                _ => "match arms have compatible types",
            },
            IfExpression => "if and else have compatible types",
            IfExpressionWithNoElse => "if missing an else returns ()",
            MainFunctionType => "`main` function has the correct type",
            StartFunctionType => "`start` function has the correct type",
            IntrinsicType => "intrinsic has the correct type",
            MethodReceiver => "method receiver has the correct type",
            _ => "types are compatible",
        }
    }
}
