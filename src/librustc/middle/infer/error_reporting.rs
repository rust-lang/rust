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
//! Having a catalogue of all the different reasons an error can arise is
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

use self::FreshOrKept::*;

use super::InferCtxt;
use super::TypeTrace;
use super::SubregionOrigin;
use super::RegionVariableOrigin;
use super::ValuePairs;
use super::region_inference::RegionResolutionError;
use super::region_inference::ConcreteFailure;
use super::region_inference::SubSupConflict;
use super::region_inference::SupSupConflict;
use super::region_inference::GenericBoundFailure;
use super::region_inference::GenericKind;
use super::region_inference::ProcessedErrors;
use super::region_inference::SameRegions;

use std::collections::HashSet;
use middle::def;
use middle::infer;
use middle::subst;
use middle::ty::{self, Ty};
use middle::ty::{Region, ReFree};
use std::cell::{Cell, RefCell};
use std::char::from_u32;
use std::rc::Rc;
use std::string::String;
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::{name_to_dummy_lifetime, PostExpansionMethod};
use syntax::owned_slice::OwnedSlice;
use syntax::codemap;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ptr::P;
use util::ppaux::bound_region_to_string;
use util::ppaux::note_and_explain_region;

// Note: only import UserString, not Repr, since user-facing error
// messages shouldn't include debug serializations.
use util::ppaux::UserString;

pub trait ErrorReporting<'tcx> {
    fn report_region_errors(&self,
                            errors: &Vec<RegionResolutionError<'tcx>>);

    fn process_errors(&self, errors: &Vec<RegionResolutionError<'tcx>>)
                      -> Vec<RegionResolutionError<'tcx>>;

    fn report_type_error(&self, trace: TypeTrace<'tcx>, terr: &ty::type_err<'tcx>);

    fn report_and_explain_type_error(&self,
                                     trace: TypeTrace<'tcx>,
                                     terr: &ty::type_err<'tcx>);

    fn values_str(&self, values: &ValuePairs<'tcx>) -> Option<String>;

    fn expected_found_str<T: UserString<'tcx> + Resolvable<'tcx>>(
        &self,
        exp_found: &ty::expected_found<T>)
        -> Option<String>;

    fn report_concrete_failure(&self,
                               origin: SubregionOrigin<'tcx>,
                               sub: Region,
                               sup: Region);

    fn report_generic_bound_failure(&self,
                                    origin: SubregionOrigin<'tcx>,
                                    kind: GenericKind<'tcx>,
                                    sub: Region,
                                    sups: Vec<Region>);

    fn report_sub_sup_conflict(&self,
                               var_origin: RegionVariableOrigin,
                               sub_origin: SubregionOrigin<'tcx>,
                               sub_region: Region,
                               sup_origin: SubregionOrigin<'tcx>,
                               sup_region: Region);

    fn report_sup_sup_conflict(&self,
                               var_origin: RegionVariableOrigin,
                               origin1: SubregionOrigin<'tcx>,
                               region1: Region,
                               origin2: SubregionOrigin<'tcx>,
                               region2: Region);

    fn report_processed_errors(&self,
                               var_origin: &[RegionVariableOrigin],
                               trace_origin: &[(TypeTrace<'tcx>, ty::type_err<'tcx>)],
                               same_regions: &[SameRegions]);

    fn give_suggestion(&self, same_regions: &[SameRegions]);
}

trait ErrorReportingHelpers<'tcx> {
    fn report_inference_failure(&self,
                                var_origin: RegionVariableOrigin);

    fn note_region_origin(&self,
                          origin: &SubregionOrigin<'tcx>);

    fn give_expl_lifetime_param(&self,
                                decl: &ast::FnDecl,
                                unsafety: ast::Unsafety,
                                ident: ast::Ident,
                                opt_explicit_self: Option<&ast::ExplicitSelf_>,
                                generics: &ast::Generics,
                                span: codemap::Span);
}

impl<'a, 'tcx> ErrorReporting<'tcx> for InferCtxt<'a, 'tcx> {
    fn report_region_errors(&self,
                            errors: &Vec<RegionResolutionError<'tcx>>) {
        let p_errors = self.process_errors(errors);
        let errors = if p_errors.is_empty() { errors } else { &p_errors };
        for error in errors {
            match error.clone() {
                ConcreteFailure(origin, sub, sup) => {
                    self.report_concrete_failure(origin, sub, sup);
                }

                GenericBoundFailure(kind, param_ty, sub, sups) => {
                    self.report_generic_bound_failure(kind, param_ty, sub, sups);
                }

                SubSupConflict(var_origin,
                               sub_origin, sub_r,
                               sup_origin, sup_r) => {
                    self.report_sub_sup_conflict(var_origin,
                                                 sub_origin, sub_r,
                                                 sup_origin, sup_r);
                }

                SupSupConflict(var_origin,
                               origin1, r1,
                               origin2, r2) => {
                    self.report_sup_sup_conflict(var_origin,
                                                 origin1, r1,
                                                 origin2, r2);
                }

                ProcessedErrors(ref var_origins,
                                ref trace_origins,
                                ref same_regions) => {
                    if !same_regions.is_empty() {
                        self.report_processed_errors(&var_origins[..],
                                                     &trace_origins[..],
                                                     &same_regions[..]);
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
    // failed (so the return value of this method should not be used)
    fn process_errors(&self, errors: &Vec<RegionResolutionError<'tcx>>)
                      -> Vec<RegionResolutionError<'tcx>> {
        debug!("process_errors()");
        let mut var_origins = Vec::new();
        let mut trace_origins = Vec::new();
        let mut same_regions = Vec::new();
        let mut processed_errors = Vec::new();
        for error in errors {
            match error.clone() {
                ConcreteFailure(origin, sub, sup) => {
                    debug!("processing ConcreteFailure");
                    let trace = match origin {
                        infer::Subtype(trace) => Some(trace),
                        _ => None,
                    };
                    match free_regions_from_same_fn(self.tcx, sub, sup) {
                        Some(ref same_frs) if trace.is_some() => {
                            let trace = trace.unwrap();
                            let terr = ty::terr_regions_does_not_outlive(sup,
                                                                         sub);
                            trace_origins.push((trace, terr));
                            append_to_same_regions(&mut same_regions, same_frs);
                        }
                        _ => processed_errors.push((*error).clone()),
                    }
                }
                SubSupConflict(var_origin, _, sub_r, _, sup_r) => {
                    debug!("processing SubSupConflict sub: {:?} sup: {:?}", sub_r, sup_r);
                    match free_regions_from_same_fn(self.tcx, sub_r, sup_r) {
                        Some(ref same_frs) => {
                            var_origins.push(var_origin);
                            append_to_same_regions(&mut same_regions, same_frs);
                        }
                        None => processed_errors.push((*error).clone()),
                    }
                }
                SupSupConflict(..) => processed_errors.push((*error).clone()),
                _ => ()  // This shouldn't happen
            }
        }
        if !same_regions.is_empty() {
            let common_scope_id = same_regions[0].scope_id;
            for sr in &same_regions {
                // Since ProcessedErrors is used to reconstruct the function
                // declaration, we want to make sure that they are, in fact,
                // from the same scope
                if sr.scope_id != common_scope_id {
                    debug!("returning empty result from process_errors because
                            {} != {}", sr.scope_id, common_scope_id);
                    return vec!();
                }
            }
            let pe = ProcessedErrors(var_origins, trace_origins, same_regions);
            debug!("errors processed: {:?}", pe);
            processed_errors.push(pe);
        }
        return processed_errors;


        struct FreeRegionsFromSameFn {
            sub_fr: ty::FreeRegion,
            sup_fr: ty::FreeRegion,
            scope_id: ast::NodeId
        }

        impl FreeRegionsFromSameFn {
            fn new(sub_fr: ty::FreeRegion,
                   sup_fr: ty::FreeRegion,
                   scope_id: ast::NodeId)
                   -> FreeRegionsFromSameFn {
                FreeRegionsFromSameFn {
                    sub_fr: sub_fr,
                    sup_fr: sup_fr,
                    scope_id: scope_id
                }
            }
        }

        fn free_regions_from_same_fn(tcx: &ty::ctxt,
                                     sub: Region,
                                     sup: Region)
                                     -> Option<FreeRegionsFromSameFn> {
            debug!("free_regions_from_same_fn(sub={:?}, sup={:?})", sub, sup);
            let (scope_id, fr1, fr2) = match (sub, sup) {
                (ReFree(fr1), ReFree(fr2)) => {
                    if fr1.scope != fr2.scope {
                        return None
                    }
                    assert!(fr1.scope == fr2.scope);
                    (fr1.scope.node_id, fr1, fr2)
                },
                _ => return None
            };
            let parent = tcx.map.get_parent(scope_id);
            let parent_node = tcx.map.find(parent);
            match parent_node {
                Some(node) => match node {
                    ast_map::NodeItem(item) => match item.node {
                        ast::ItemFn(..) => {
                            Some(FreeRegionsFromSameFn::new(fr1, fr2, scope_id))
                        },
                        _ => None
                    },
                    ast_map::NodeImplItem(..) |
                    ast_map::NodeTraitItem(..) => {
                        Some(FreeRegionsFromSameFn::new(fr1, fr2, scope_id))
                    },
                    _ => None
                },
                None => {
                    debug!("no parent node of scope_id {}", scope_id);
                    None
                }
            }
        }

        fn append_to_same_regions(same_regions: &mut Vec<SameRegions>,
                                  same_frs: &FreeRegionsFromSameFn) {
            let scope_id = same_frs.scope_id;
            let (sub_fr, sup_fr) = (same_frs.sub_fr, same_frs.sup_fr);
            for sr in &mut *same_regions {
                if sr.contains(&sup_fr.bound_region)
                   && scope_id == sr.scope_id {
                    sr.push(sub_fr.bound_region);
                    return
                }
            }
            same_regions.push(SameRegions {
                scope_id: scope_id,
                regions: vec!(sub_fr.bound_region, sup_fr.bound_region)
            })
        }
    }

    fn report_type_error(&self, trace: TypeTrace<'tcx>, terr: &ty::type_err<'tcx>) {
        let expected_found_str = match self.values_str(&trace.values) {
            Some(v) => v,
            None => {
                return; /* derived error */
            }
        };

        let message_root_str = match trace.origin {
            infer::Misc(_) => "mismatched types",
            infer::MethodCompatCheck(_) => "method not compatible with trait",
            infer::ExprAssignable(_) => "mismatched types",
            infer::RelateTraitRefs(_) => "mismatched traits",
            infer::RelateSelfType(_) => "mismatched types",
            infer::RelateOutputImplTypes(_) => "mismatched types",
            infer::MatchExpressionArm(_, _) => "match arms have incompatible types",
            infer::IfExpression(_) => "if and else have incompatible types",
            infer::IfExpressionWithNoElse(_) => "if may be missing an else clause",
            infer::RangeExpression(_) => "start and end of range have incompatible types",
            infer::EquatePredicate(_) => "equality predicate not satisfied",
        };

        span_err!(self.tcx.sess, trace.origin.span(), E0308,
            "{}: {} ({})",
                 message_root_str,
                 expected_found_str,
                 ty::type_err_to_str(self.tcx, terr));

        match trace.origin {
            infer::MatchExpressionArm(_, arm_span) =>
                self.tcx.sess.span_note(arm_span, "match arm with an incompatible type"),
            _ => ()
        }
    }

    fn report_and_explain_type_error(&self,
                                     trace: TypeTrace<'tcx>,
                                     terr: &ty::type_err<'tcx>) {
        self.report_type_error(trace, terr);
        ty::note_and_explain_type_err(self.tcx, terr);
    }

    /// Returns a string of the form "expected `{}`, found `{}`", or None if this is a derived
    /// error.
    fn values_str(&self, values: &ValuePairs<'tcx>) -> Option<String> {
        match *values {
            infer::Types(ref exp_found) => self.expected_found_str(exp_found),
            infer::TraitRefs(ref exp_found) => self.expected_found_str(exp_found),
            infer::PolyTraitRefs(ref exp_found) => self.expected_found_str(exp_found)
        }
    }

    fn expected_found_str<T: UserString<'tcx> + Resolvable<'tcx>>(
        &self,
        exp_found: &ty::expected_found<T>)
        -> Option<String>
    {
        let expected = exp_found.expected.resolve(self);
        if expected.contains_error() {
            return None;
        }

        let found = exp_found.found.resolve(self);
        if found.contains_error() {
            return None;
        }

        Some(format!("expected `{}`, found `{}`",
                     expected.user_string(self.tcx),
                     found.user_string(self.tcx)))
    }

    fn report_generic_bound_failure(&self,
                                    origin: SubregionOrigin<'tcx>,
                                    bound_kind: GenericKind<'tcx>,
                                    sub: Region,
                                    _sups: Vec<Region>)
    {
        // FIXME: it would be better to report the first error message
        // with the span of the parameter itself, rather than the span
        // where the error was detected. But that span is not readily
        // accessible.

        let labeled_user_string = match bound_kind {
            GenericKind::Param(ref p) =>
                format!("the parameter type `{}`", p.user_string(self.tcx)),
            GenericKind::Projection(ref p) =>
                format!("the associated type `{}`", p.user_string(self.tcx)),
        };

        match sub {
            ty::ReFree(ty::FreeRegion {bound_region: ty::BrNamed(..), ..}) => {
                // Does the required lifetime have a nice name we can print?
                span_err!(self.tcx.sess, origin.span(), E0309,
                    "{} may not live long enough", labeled_user_string);
                self.tcx.sess.span_help(
                    origin.span(),
                    &format!(
                        "consider adding an explicit lifetime bound `{}: {}`...",
                        bound_kind.user_string(self.tcx),
                        sub.user_string(self.tcx))[]);
            }

            ty::ReStatic => {
                // Does the required lifetime have a nice name we can print?
                span_err!(self.tcx.sess, origin.span(), E0310,
                    "{} may not live long enough", labeled_user_string);
                self.tcx.sess.span_help(
                    origin.span(),
                    &format!(
                        "consider adding an explicit lifetime bound `{}: 'static`...",
                        bound_kind.user_string(self.tcx))[]);
            }

            _ => {
                // If not, be less specific.
                span_err!(self.tcx.sess, origin.span(), E0311,
                        "{} may not live long enough",
                        labeled_user_string);
                self.tcx.sess.span_help(
                    origin.span(),
                    &format!(
                        "consider adding an explicit lifetime bound for `{}`",
                        bound_kind.user_string(self.tcx))[]);
                note_and_explain_region(
                    self.tcx,
                    &format!("{} must be valid for ", labeled_user_string)[],
                    sub,
                    "...");
            }
        }

        self.note_region_origin(&origin);
    }

    fn report_concrete_failure(&self,
                               origin: SubregionOrigin<'tcx>,
                               sub: Region,
                               sup: Region) {
        match origin {
            infer::Subtype(trace) => {
                let terr = ty::terr_regions_does_not_outlive(sup, sub);
                self.report_and_explain_type_error(trace, &terr);
            }
            infer::Reborrow(span) => {
                span_err!(self.tcx.sess, span, E0312,
                    "lifetime of reference outlines \
                     lifetime of borrowed content...");
                note_and_explain_region(
                    self.tcx,
                    "...the reference is valid for ",
                    sub,
                    "...");
                note_and_explain_region(
                    self.tcx,
                    "...but the borrowed content is only valid for ",
                    sup,
                    "");
            }
            infer::ReborrowUpvar(span, ref upvar_id) => {
                span_err!(self.tcx.sess, span, E0313,
                    "lifetime of borrowed pointer outlives \
                            lifetime of captured variable `{}`...",
                            ty::local_var_name_str(self.tcx,
                                                   upvar_id.var_id)
                                .to_string());
                note_and_explain_region(
                    self.tcx,
                    "...the borrowed pointer is valid for ",
                    sub,
                    "...");
                note_and_explain_region(
                    self.tcx,
                    &format!("...but `{}` is only valid for ",
                            ty::local_var_name_str(self.tcx,
                                                   upvar_id.var_id)
                                .to_string())[],
                    sup,
                    "");
            }
            infer::InfStackClosure(span) => {
                span_err!(self.tcx.sess, span, E0314,
                    "closure outlives stack frame");
                note_and_explain_region(
                    self.tcx,
                    "...the closure must be valid for ",
                    sub,
                    "...");
                note_and_explain_region(
                    self.tcx,
                    "...but the closure's stack frame is only valid for ",
                    sup,
                    "");
            }
            infer::InvokeClosure(span) => {
                span_err!(self.tcx.sess, span, E0315,
                    "cannot invoke closure outside of its lifetime");
                note_and_explain_region(
                    self.tcx,
                    "the closure is only valid for ",
                    sup,
                    "");
            }
            infer::DerefPointer(span) => {
                self.tcx.sess.span_err(
                    span,
                    "dereference of reference outside its lifetime");
                note_and_explain_region(
                    self.tcx,
                    "the reference is only valid for ",
                    sup,
                    "");
            }
            infer::FreeVariable(span, id) => {
                self.tcx.sess.span_err(
                    span,
                    &format!("captured variable `{}` does not \
                            outlive the enclosing closure",
                            ty::local_var_name_str(self.tcx,
                                                   id).to_string())[]);
                note_and_explain_region(
                    self.tcx,
                    "captured variable is valid for ",
                    sup,
                    "");
                note_and_explain_region(
                    self.tcx,
                    "closure is valid for ",
                    sub,
                    "");
            }
            infer::IndexSlice(span) => {
                self.tcx.sess.span_err(span,
                                       "index of slice outside its lifetime");
                note_and_explain_region(
                    self.tcx,
                    "the slice is only valid for ",
                    sup,
                    "");
            }
            infer::RelateObjectBound(span) => {
                self.tcx.sess.span_err(
                    span,
                    "lifetime of the source pointer does not outlive \
                     lifetime bound of the object type");
                note_and_explain_region(
                    self.tcx,
                    "object type is valid for ",
                    sub,
                    "");
                note_and_explain_region(
                    self.tcx,
                    "source pointer is only valid for ",
                    sup,
                    "");
            }
            infer::RelateParamBound(span, ty) => {
                self.tcx.sess.span_err(
                    span,
                    &format!("the type `{}` does not fulfill the \
                             required lifetime",
                            self.ty_to_string(ty))[]);
                note_and_explain_region(self.tcx,
                                        "type must outlive ",
                                        sub,
                                        "");
            }
            infer::RelateRegionParamBound(span) => {
                self.tcx.sess.span_err(
                    span,
                    "lifetime bound not satisfied");
                note_and_explain_region(
                    self.tcx,
                    "lifetime parameter instantiated with ",
                    sup,
                    "");
                note_and_explain_region(
                    self.tcx,
                    "but lifetime parameter must outlive ",
                    sub,
                    "");
            }
            infer::RelateDefaultParamBound(span, ty) => {
                self.tcx.sess.span_err(
                    span,
                    &format!("the type `{}` (provided as the value of \
                             a type parameter) is not valid at this point",
                            self.ty_to_string(ty))[]);
                note_and_explain_region(self.tcx,
                                        "type must outlive ",
                                        sub,
                                        "");
            }
            infer::CallRcvr(span) => {
                self.tcx.sess.span_err(
                    span,
                    "lifetime of method receiver does not outlive \
                     the method call");
                note_and_explain_region(
                    self.tcx,
                    "the receiver is only valid for ",
                    sup,
                    "");
            }
            infer::CallArg(span) => {
                self.tcx.sess.span_err(
                    span,
                    "lifetime of function argument does not outlive \
                     the function call");
                note_and_explain_region(
                    self.tcx,
                    "the function argument is only valid for ",
                    sup,
                    "");
            }
            infer::CallReturn(span) => {
                self.tcx.sess.span_err(
                    span,
                    "lifetime of return value does not outlive \
                     the function call");
                note_and_explain_region(
                    self.tcx,
                    "the return value is only valid for ",
                    sup,
                    "");
            }
            infer::Operand(span) => {
                self.tcx.sess.span_err(
                    span,
                    "lifetime of operand does not outlive \
                     the operation");
                note_and_explain_region(
                    self.tcx,
                    "the operand is only valid for ",
                    sup,
                    "");
            }
            infer::AddrOf(span) => {
                self.tcx.sess.span_err(
                    span,
                    "reference is not valid \
                     at the time of borrow");
                note_and_explain_region(
                    self.tcx,
                    "the borrow is only valid for ",
                    sup,
                    "");
            }
            infer::AutoBorrow(span) => {
                self.tcx.sess.span_err(
                    span,
                    "automatically reference is not valid \
                     at the time of borrow");
                note_and_explain_region(
                    self.tcx,
                    "the automatic borrow is only valid for ",
                    sup,
                    "");
            }
            infer::ExprTypeIsNotInScope(t, span) => {
                self.tcx.sess.span_err(
                    span,
                    &format!("type of expression contains references \
                             that are not valid during the expression: `{}`",
                            self.ty_to_string(t))[]);
                note_and_explain_region(
                    self.tcx,
                    "type is only valid for ",
                    sup,
                    "");
            }
            infer::SafeDestructor(span) => {
                self.tcx.sess.span_err(
                    span,
                    "unsafe use of destructor: destructor might be called \
                     while references are dead");
                // FIXME (22171): terms "super/subregion" are suboptimal
                note_and_explain_region(
                    self.tcx,
                    "superregion: ",
                    sup,
                    "");
                note_and_explain_region(
                    self.tcx,
                    "subregion: ",
                    sub,
                    "");
            }
            infer::BindingTypeIsNotValidAtDecl(span) => {
                self.tcx.sess.span_err(
                    span,
                    "lifetime of variable does not enclose its declaration");
                note_and_explain_region(
                    self.tcx,
                    "the variable is only valid for ",
                    sup,
                    "");
            }
            infer::ReferenceOutlivesReferent(ty, span) => {
                self.tcx.sess.span_err(
                    span,
                    &format!("in type `{}`, reference has a longer lifetime \
                             than the data it references",
                            self.ty_to_string(ty))[]);
                note_and_explain_region(
                    self.tcx,
                    "the pointer is valid for ",
                    sub,
                    "");
                note_and_explain_region(
                    self.tcx,
                    "but the referenced data is only valid for ",
                    sup,
                    "");
            }
        }
    }

    fn report_sub_sup_conflict(&self,
                               var_origin: RegionVariableOrigin,
                               sub_origin: SubregionOrigin<'tcx>,
                               sub_region: Region,
                               sup_origin: SubregionOrigin<'tcx>,
                               sup_region: Region) {
        self.report_inference_failure(var_origin);

        note_and_explain_region(
            self.tcx,
            "first, the lifetime cannot outlive ",
            sup_region,
            "...");

        self.note_region_origin(&sup_origin);

        note_and_explain_region(
            self.tcx,
            "but, the lifetime must be valid for ",
            sub_region,
            "...");

        self.note_region_origin(&sub_origin);
    }

    fn report_sup_sup_conflict(&self,
                               var_origin: RegionVariableOrigin,
                               origin1: SubregionOrigin<'tcx>,
                               region1: Region,
                               origin2: SubregionOrigin<'tcx>,
                               region2: Region) {
        self.report_inference_failure(var_origin);

        note_and_explain_region(
            self.tcx,
            "first, the lifetime must be contained by ",
            region1,
            "...");

        self.note_region_origin(&origin1);

        note_and_explain_region(
            self.tcx,
            "but, the lifetime must also be contained by ",
            region2,
            "...");

        self.note_region_origin(&origin2);
    }

    fn report_processed_errors(&self,
                               var_origins: &[RegionVariableOrigin],
                               trace_origins: &[(TypeTrace<'tcx>, ty::type_err<'tcx>)],
                               same_regions: &[SameRegions]) {
        for vo in var_origins {
            self.report_inference_failure(vo.clone());
        }
        self.give_suggestion(same_regions);
        for &(ref trace, terr) in trace_origins {
            self.report_type_error(trace.clone(), &terr);
        }
    }

    fn give_suggestion(&self, same_regions: &[SameRegions]) {
        let scope_id = same_regions[0].scope_id;
        let parent = self.tcx.map.get_parent(scope_id);
        let parent_node = self.tcx.map.find(parent);
        let taken = lifetimes_in_scope(self.tcx, scope_id);
        let life_giver = LifeGiver::with_taken(&taken[..]);
        let node_inner = match parent_node {
            Some(ref node) => match *node {
                ast_map::NodeItem(ref item) => {
                    match item.node {
                        ast::ItemFn(ref fn_decl, pur, _, ref gen, _) => {
                            Some((&**fn_decl, gen, pur, item.ident, None, item.span))
                        },
                        _ => None
                    }
                }
                ast_map::NodeImplItem(ref item) => {
                    match **item {
                        ast::MethodImplItem(ref m) => {
                            Some((m.pe_fn_decl(),
                                  m.pe_generics(),
                                  m.pe_unsafety(),
                                  m.pe_ident(),
                                  Some(&m.pe_explicit_self().node),
                                  m.span))
                        }
                        ast::TypeImplItem(_) => None,
                    }
                },
                ast_map::NodeTraitItem(ref item) => {
                    match **item {
                        ast::ProvidedMethod(ref m) => {
                            Some((m.pe_fn_decl(),
                                  m.pe_generics(),
                                  m.pe_unsafety(),
                                  m.pe_ident(),
                                  Some(&m.pe_explicit_self().node),
                                  m.span))
                        }
                        _ => None
                    }
                }
                _ => None
            },
            None => None
        };
        let (fn_decl, generics, unsafety, ident, expl_self, span)
                                    = node_inner.expect("expect item fn");
        let rebuilder = Rebuilder::new(self.tcx, fn_decl, expl_self,
                                       generics, same_regions, &life_giver);
        let (fn_decl, expl_self, generics) = rebuilder.rebuild();
        self.give_expl_lifetime_param(&fn_decl, unsafety, ident,
                                      expl_self.as_ref(), &generics, span);
    }
}

struct RebuildPathInfo<'a> {
    path: &'a ast::Path,
    // indexes to insert lifetime on path.lifetimes
    indexes: Vec<u32>,
    // number of lifetimes we expect to see on the type referred by `path`
    // (e.g., expected=1 for struct Foo<'a>)
    expected: u32,
    anon_nums: &'a HashSet<u32>,
    region_names: &'a HashSet<ast::Name>
}

struct Rebuilder<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    fn_decl: &'a ast::FnDecl,
    expl_self_opt: Option<&'a ast::ExplicitSelf_>,
    generics: &'a ast::Generics,
    same_regions: &'a [SameRegions],
    life_giver: &'a LifeGiver,
    cur_anon: Cell<u32>,
    inserted_anons: RefCell<HashSet<u32>>,
}

enum FreshOrKept {
    Fresh,
    Kept
}

impl<'a, 'tcx> Rebuilder<'a, 'tcx> {
    fn new(tcx: &'a ty::ctxt<'tcx>,
           fn_decl: &'a ast::FnDecl,
           expl_self_opt: Option<&'a ast::ExplicitSelf_>,
           generics: &'a ast::Generics,
           same_regions: &'a [SameRegions],
           life_giver: &'a LifeGiver)
           -> Rebuilder<'a, 'tcx> {
        Rebuilder {
            tcx: tcx,
            fn_decl: fn_decl,
            expl_self_opt: expl_self_opt,
            generics: generics,
            same_regions: same_regions,
            life_giver: life_giver,
            cur_anon: Cell::new(0),
            inserted_anons: RefCell::new(HashSet::new()),
        }
    }

    fn rebuild(&self)
               -> (ast::FnDecl, Option<ast::ExplicitSelf_>, ast::Generics) {
        let mut expl_self_opt = self.expl_self_opt.cloned();
        let mut inputs = self.fn_decl.inputs.clone();
        let mut output = self.fn_decl.output.clone();
        let mut ty_params = self.generics.ty_params.clone();
        let where_clause = self.generics.where_clause.clone();
        let mut kept_lifetimes = HashSet::new();
        for sr in self.same_regions {
            self.cur_anon.set(0);
            self.offset_cur_anon();
            let (anon_nums, region_names) =
                                self.extract_anon_nums_and_names(sr);
            let (lifetime, fresh_or_kept) = self.pick_lifetime(&region_names);
            match fresh_or_kept {
                Kept => { kept_lifetimes.insert(lifetime.name); }
                _ => ()
            }
            expl_self_opt = self.rebuild_expl_self(expl_self_opt, lifetime,
                                                   &anon_nums, &region_names);
            inputs = self.rebuild_args_ty(&inputs[..], lifetime,
                                          &anon_nums, &region_names);
            output = self.rebuild_output(&output, lifetime, &anon_nums, &region_names);
            ty_params = self.rebuild_ty_params(ty_params, lifetime,
                                               &region_names);
        }
        let fresh_lifetimes = self.life_giver.get_generated_lifetimes();
        let all_region_names = self.extract_all_region_names();
        let generics = self.rebuild_generics(self.generics,
                                             &fresh_lifetimes,
                                             &kept_lifetimes,
                                             &all_region_names,
                                             ty_params,
                                             where_clause);
        let new_fn_decl = ast::FnDecl {
            inputs: inputs,
            output: output,
            variadic: self.fn_decl.variadic
        };
        (new_fn_decl, expl_self_opt, generics)
    }

    fn pick_lifetime(&self,
                     region_names: &HashSet<ast::Name>)
                     -> (ast::Lifetime, FreshOrKept) {
        if region_names.len() > 0 {
            // It's not necessary to convert the set of region names to a
            // vector of string and then sort them. However, it makes the
            // choice of lifetime name deterministic and thus easier to test.
            let mut names = Vec::new();
            for rn in region_names {
                let lt_name = token::get_name(*rn).to_string();
                names.push(lt_name);
            }
            names.sort();
            let name = token::str_to_ident(&names[0][]).name;
            return (name_to_dummy_lifetime(name), Kept);
        }
        return (self.life_giver.give_lifetime(), Fresh);
    }

    fn extract_anon_nums_and_names(&self, same_regions: &SameRegions)
                                   -> (HashSet<u32>, HashSet<ast::Name>) {
        let mut anon_nums = HashSet::new();
        let mut region_names = HashSet::new();
        for br in &same_regions.regions {
            match *br {
                ty::BrAnon(i) => {
                    anon_nums.insert(i);
                }
                ty::BrNamed(_, name) => {
                    region_names.insert(name);
                }
                _ => ()
            }
        }
        (anon_nums, region_names)
    }

    fn extract_all_region_names(&self) -> HashSet<ast::Name> {
        let mut all_region_names = HashSet::new();
        for sr in self.same_regions {
            for br in &sr.regions {
                match *br {
                    ty::BrNamed(_, name) => {
                        all_region_names.insert(name);
                    }
                    _ => ()
                }
            }
        }
        all_region_names
    }

    fn inc_cur_anon(&self, n: u32) {
        let anon = self.cur_anon.get();
        self.cur_anon.set(anon+n);
    }

    fn offset_cur_anon(&self) {
        let mut anon = self.cur_anon.get();
        while self.inserted_anons.borrow().contains(&anon) {
            anon += 1;
        }
        self.cur_anon.set(anon);
    }

    fn inc_and_offset_cur_anon(&self, n: u32) {
        self.inc_cur_anon(n);
        self.offset_cur_anon();
    }

    fn track_anon(&self, anon: u32) {
        self.inserted_anons.borrow_mut().insert(anon);
    }

    fn rebuild_ty_params(&self,
                         ty_params: OwnedSlice<ast::TyParam>,
                         lifetime: ast::Lifetime,
                         region_names: &HashSet<ast::Name>)
                         -> OwnedSlice<ast::TyParam> {
        ty_params.map(|ty_param| {
            let bounds = self.rebuild_ty_param_bounds(ty_param.bounds.clone(),
                                                      lifetime,
                                                      region_names);
            ast::TyParam {
                ident: ty_param.ident,
                id: ty_param.id,
                bounds: bounds,
                default: ty_param.default.clone(),
                span: ty_param.span,
            }
        })
    }

    fn rebuild_ty_param_bounds(&self,
                               ty_param_bounds: OwnedSlice<ast::TyParamBound>,
                               lifetime: ast::Lifetime,
                               region_names: &HashSet<ast::Name>)
                               -> OwnedSlice<ast::TyParamBound> {
        ty_param_bounds.map(|tpb| {
            match tpb {
                &ast::RegionTyParamBound(lt) => {
                    // FIXME -- it's unclear whether I'm supposed to
                    // substitute lifetime here. I suspect we need to
                    // be passing down a map.
                    ast::RegionTyParamBound(lt)
                }
                &ast::TraitTyParamBound(ref poly_tr, modifier) => {
                    let tr = &poly_tr.trait_ref;
                    let last_seg = tr.path.segments.last().unwrap();
                    let mut insert = Vec::new();
                    let lifetimes = last_seg.parameters.lifetimes();
                    for (i, lt) in lifetimes.iter().enumerate() {
                        if region_names.contains(&lt.name) {
                            insert.push(i as u32);
                        }
                    }
                    let rebuild_info = RebuildPathInfo {
                        path: &tr.path,
                        indexes: insert,
                        expected: lifetimes.len() as u32,
                        anon_nums: &HashSet::new(),
                        region_names: region_names
                    };
                    let new_path = self.rebuild_path(rebuild_info, lifetime);
                    ast::TraitTyParamBound(ast::PolyTraitRef {
                        bound_lifetimes: poly_tr.bound_lifetimes.clone(),
                        trait_ref: ast::TraitRef {
                            path: new_path,
                            ref_id: tr.ref_id,
                        },
                        span: poly_tr.span,
                    }, modifier)
                }
            }
        })
    }

    fn rebuild_expl_self(&self,
                         expl_self_opt: Option<ast::ExplicitSelf_>,
                         lifetime: ast::Lifetime,
                         anon_nums: &HashSet<u32>,
                         region_names: &HashSet<ast::Name>)
                         -> Option<ast::ExplicitSelf_> {
        match expl_self_opt {
            Some(ref expl_self) => match *expl_self {
                ast::SelfRegion(lt_opt, muta, id) => match lt_opt {
                    Some(lt) => if region_names.contains(&lt.name) {
                        return Some(ast::SelfRegion(Some(lifetime), muta, id));
                    },
                    None => {
                        let anon = self.cur_anon.get();
                        self.inc_and_offset_cur_anon(1);
                        if anon_nums.contains(&anon) {
                            self.track_anon(anon);
                            return Some(ast::SelfRegion(Some(lifetime), muta, id));
                        }
                    }
                },
                _ => ()
            },
            None => ()
        }
        expl_self_opt
    }

    fn rebuild_generics(&self,
                        generics: &ast::Generics,
                        add: &Vec<ast::Lifetime>,
                        keep: &HashSet<ast::Name>,
                        remove: &HashSet<ast::Name>,
                        ty_params: OwnedSlice<ast::TyParam>,
                        where_clause: ast::WhereClause)
                        -> ast::Generics {
        let mut lifetimes = Vec::new();
        for lt in add {
            lifetimes.push(ast::LifetimeDef { lifetime: *lt,
                                              bounds: Vec::new() });
        }
        for lt in &generics.lifetimes {
            if keep.contains(&lt.lifetime.name) ||
                !remove.contains(&lt.lifetime.name) {
                lifetimes.push((*lt).clone());
            }
        }
        ast::Generics {
            lifetimes: lifetimes,
            ty_params: ty_params,
            where_clause: where_clause,
        }
    }

    fn rebuild_args_ty(&self,
                       inputs: &[ast::Arg],
                       lifetime: ast::Lifetime,
                       anon_nums: &HashSet<u32>,
                       region_names: &HashSet<ast::Name>)
                       -> Vec<ast::Arg> {
        let mut new_inputs = Vec::new();
        for arg in inputs {
            let new_ty = self.rebuild_arg_ty_or_output(&*arg.ty, lifetime,
                                                       anon_nums, region_names);
            let possibly_new_arg = ast::Arg {
                ty: new_ty,
                pat: arg.pat.clone(),
                id: arg.id
            };
            new_inputs.push(possibly_new_arg);
        }
        new_inputs
    }

    fn rebuild_output(&self, ty: &ast::FunctionRetTy,
                      lifetime: ast::Lifetime,
                      anon_nums: &HashSet<u32>,
                      region_names: &HashSet<ast::Name>) -> ast::FunctionRetTy {
        match *ty {
            ast::Return(ref ret_ty) => ast::Return(
                self.rebuild_arg_ty_or_output(&**ret_ty, lifetime, anon_nums, region_names)
            ),
            ast::DefaultReturn(span) => ast::DefaultReturn(span),
            ast::NoReturn(span) => ast::NoReturn(span)
        }
    }

    fn rebuild_arg_ty_or_output(&self,
                                ty: &ast::Ty,
                                lifetime: ast::Lifetime,
                                anon_nums: &HashSet<u32>,
                                region_names: &HashSet<ast::Name>)
                                -> P<ast::Ty> {
        let mut new_ty = P(ty.clone());
        let mut ty_queue = vec!(ty);
        while !ty_queue.is_empty() {
            let cur_ty = ty_queue.remove(0);
            match cur_ty.node {
                ast::TyRptr(lt_opt, ref mut_ty) => {
                    let rebuild = match lt_opt {
                        Some(lt) => region_names.contains(&lt.name),
                        None => {
                            let anon = self.cur_anon.get();
                            let rebuild = anon_nums.contains(&anon);
                            if rebuild {
                                self.track_anon(anon);
                            }
                            self.inc_and_offset_cur_anon(1);
                            rebuild
                        }
                    };
                    if rebuild {
                        let to = ast::Ty {
                            id: cur_ty.id,
                            node: ast::TyRptr(Some(lifetime), mut_ty.clone()),
                            span: cur_ty.span
                        };
                        new_ty = self.rebuild_ty(new_ty, P(to));
                    }
                    ty_queue.push(&*mut_ty.ty);
                }
                ast::TyPath(ref path, id) => {
                    let a_def = match self.tcx.def_map.borrow().get(&id) {
                        None => {
                            self.tcx
                                .sess
                                .fatal(&format!(
                                        "unbound path {}",
                                        pprust::path_to_string(path))[])
                        }
                        Some(&d) => d
                    };
                    match a_def {
                        def::DefTy(did, _) | def::DefStruct(did) => {
                            let generics = ty::lookup_item_type(self.tcx, did).generics;

                            let expected =
                                generics.regions.len(subst::TypeSpace) as u32;
                            let lifetimes =
                                path.segments.last().unwrap().parameters.lifetimes();
                            let mut insert = Vec::new();
                            if lifetimes.len() == 0 {
                                let anon = self.cur_anon.get();
                                for (i, a) in (anon..anon+expected).enumerate() {
                                    if anon_nums.contains(&a) {
                                        insert.push(i as u32);
                                    }
                                    self.track_anon(a);
                                }
                                self.inc_and_offset_cur_anon(expected);
                            } else {
                                for (i, lt) in lifetimes.iter().enumerate() {
                                    if region_names.contains(&lt.name) {
                                        insert.push(i as u32);
                                    }
                                }
                            }
                            let rebuild_info = RebuildPathInfo {
                                path: path,
                                indexes: insert,
                                expected: expected,
                                anon_nums: anon_nums,
                                region_names: region_names
                            };
                            let new_path = self.rebuild_path(rebuild_info, lifetime);
                            let to = ast::Ty {
                                id: cur_ty.id,
                                node: ast::TyPath(new_path, id),
                                span: cur_ty.span
                            };
                            new_ty = self.rebuild_ty(new_ty, P(to));
                        }
                        _ => ()
                    }

                }

                ast::TyPtr(ref mut_ty) => {
                    ty_queue.push(&*mut_ty.ty);
                }
                ast::TyVec(ref ty) |
                ast::TyFixedLengthVec(ref ty, _) => {
                    ty_queue.push(&**ty);
                }
                ast::TyTup(ref tys) => ty_queue.extend(tys.iter().map(|ty| &**ty)),
                _ => {}
            }
        }
        new_ty
    }

    fn rebuild_ty(&self,
                  from: P<ast::Ty>,
                  to: P<ast::Ty>)
                  -> P<ast::Ty> {

        fn build_to(from: P<ast::Ty>,
                    to: &mut Option<P<ast::Ty>>)
                    -> P<ast::Ty> {
            if Some(from.id) == to.as_ref().map(|ty| ty.id) {
                return to.take().expect("`to` type found more than once during rebuild");
            }
            from.map(|ast::Ty {id, node, span}| {
                let new_node = match node {
                    ast::TyRptr(lifetime, mut_ty) => {
                        ast::TyRptr(lifetime, ast::MutTy {
                            mutbl: mut_ty.mutbl,
                            ty: build_to(mut_ty.ty, to),
                        })
                    }
                    ast::TyPtr(mut_ty) => {
                        ast::TyPtr(ast::MutTy {
                            mutbl: mut_ty.mutbl,
                            ty: build_to(mut_ty.ty, to),
                        })
                    }
                    ast::TyVec(ty) => ast::TyVec(build_to(ty, to)),
                    ast::TyFixedLengthVec(ty, e) => {
                        ast::TyFixedLengthVec(build_to(ty, to), e)
                    }
                    ast::TyTup(tys) => {
                        ast::TyTup(tys.into_iter().map(|ty| build_to(ty, to)).collect())
                    }
                    ast::TyParen(typ) => ast::TyParen(build_to(typ, to)),
                    other => other
                };
                ast::Ty { id: id, node: new_node, span: span }
            })
        }

        build_to(from, &mut Some(to))
    }

    fn rebuild_path(&self,
                    rebuild_info: RebuildPathInfo,
                    lifetime: ast::Lifetime)
                    -> ast::Path
    {
        let RebuildPathInfo {
            path,
            indexes,
            expected,
            anon_nums,
            region_names,
        } = rebuild_info;

        let last_seg = path.segments.last().unwrap();
        let new_parameters = match last_seg.parameters {
            ast::ParenthesizedParameters(..) => {
                last_seg.parameters.clone()
            }

            ast::AngleBracketedParameters(ref data) => {
                let mut new_lts = Vec::new();
                if data.lifetimes.len() == 0 {
                    // traverse once to see if there's a need to insert lifetime
                    let need_insert = (0..expected).any(|i| {
                        indexes.contains(&i)
                    });
                    if need_insert {
                        for i in 0..expected {
                            if indexes.contains(&i) {
                                new_lts.push(lifetime);
                            } else {
                                new_lts.push(self.life_giver.give_lifetime());
                            }
                        }
                    }
                } else {
                    for (i, lt) in data.lifetimes.iter().enumerate() {
                        if indexes.contains(&(i as u32)) {
                            new_lts.push(lifetime);
                        } else {
                            new_lts.push(*lt);
                        }
                    }
                }
                let new_types = data.types.map(|t| {
                    self.rebuild_arg_ty_or_output(&**t, lifetime, anon_nums, region_names)
                });
                let new_bindings = data.bindings.map(|b| {
                    P(ast::TypeBinding {
                        id: b.id,
                        ident: b.ident,
                        ty: self.rebuild_arg_ty_or_output(&*b.ty,
                                                          lifetime,
                                                          anon_nums,
                                                          region_names),
                        span: b.span
                    })
                });
                ast::AngleBracketedParameters(ast::AngleBracketedParameterData {
                    lifetimes: new_lts,
                    types: new_types,
                    bindings: new_bindings,
               })
            }
        };
        let new_seg = ast::PathSegment {
            identifier: last_seg.identifier,
            parameters: new_parameters
        };
        let mut new_segs = Vec::new();
        new_segs.push_all(path.segments.init());
        new_segs.push(new_seg);
        ast::Path {
            span: path.span,
            global: path.global,
            segments: new_segs
        }
    }
}

impl<'a, 'tcx> ErrorReportingHelpers<'tcx> for InferCtxt<'a, 'tcx> {
    fn give_expl_lifetime_param(&self,
                                decl: &ast::FnDecl,
                                unsafety: ast::Unsafety,
                                ident: ast::Ident,
                                opt_explicit_self: Option<&ast::ExplicitSelf_>,
                                generics: &ast::Generics,
                                span: codemap::Span) {
        let suggested_fn = pprust::fun_to_string(decl, unsafety, ident,
                                              opt_explicit_self, generics);
        let msg = format!("consider using an explicit lifetime \
                           parameter as shown: {}", suggested_fn);
        self.tcx.sess.span_help(span, &msg[..]);
    }

    fn report_inference_failure(&self,
                                var_origin: RegionVariableOrigin) {
        let var_description = match var_origin {
            infer::MiscVariable(_) => "".to_string(),
            infer::PatternRegion(_) => " for pattern".to_string(),
            infer::AddrOfRegion(_) => " for borrow expression".to_string(),
            infer::Autoref(_) => " for autoref".to_string(),
            infer::Coercion(_) => " for automatic coercion".to_string(),
            infer::LateBoundRegion(_, br, infer::FnCall) => {
                format!(" for {}in function call",
                        bound_region_to_string(self.tcx, "lifetime parameter ", true, br))
            }
            infer::LateBoundRegion(_, br, infer::HigherRankedType) => {
                format!(" for {}in generic type",
                        bound_region_to_string(self.tcx, "lifetime parameter ", true, br))
            }
            infer::LateBoundRegion(_, br, infer::AssocTypeProjection(type_name)) => {
                format!(" for {}in trait containing associated type `{}`",
                        bound_region_to_string(self.tcx, "lifetime parameter ", true, br),
                        token::get_name(type_name))
            }
            infer::EarlyBoundRegion(_, name) => {
                format!(" for lifetime parameter `{}`",
                        &token::get_name(name))
            }
            infer::BoundRegionInCoherence(name) => {
                format!(" for lifetime parameter `{}` in coherence check",
                        &token::get_name(name))
            }
            infer::UpvarRegion(ref upvar_id, _) => {
                format!(" for capture of `{}` by closure",
                        ty::local_var_name_str(self.tcx, upvar_id.var_id).to_string())
            }
        };

        self.tcx.sess.span_err(
            var_origin.span(),
            &format!("cannot infer an appropriate lifetime{} \
                    due to conflicting requirements",
                    var_description)[]);
    }

    fn note_region_origin(&self, origin: &SubregionOrigin<'tcx>) {
        match *origin {
            infer::Subtype(ref trace) => {
                let desc = match trace.origin {
                    infer::Misc(_) => {
                        format!("types are compatible")
                    }
                    infer::MethodCompatCheck(_) => {
                        format!("method type is compatible with trait")
                    }
                    infer::ExprAssignable(_) => {
                        format!("expression is assignable")
                    }
                    infer::RelateTraitRefs(_) => {
                        format!("traits are compatible")
                    }
                    infer::RelateSelfType(_) => {
                        format!("self type matches impl self type")
                    }
                    infer::RelateOutputImplTypes(_) => {
                        format!("trait type parameters matches those \
                                 specified on the impl")
                    }
                    infer::MatchExpressionArm(_, _) => {
                        format!("match arms have compatible types")
                    }
                    infer::IfExpression(_) => {
                        format!("if and else have compatible types")
                    }
                    infer::IfExpressionWithNoElse(_) => {
                        format!("if may be missing an else clause")
                    }
                    infer::RangeExpression(_) => {
                        format!("start and end of range have compatible types")
                    }
                    infer::EquatePredicate(_) => {
                        format!("equality where clause is satisfied")
                    }
                };

                match self.values_str(&trace.values) {
                    Some(values_str) => {
                        self.tcx.sess.span_note(
                            trace.origin.span(),
                            &format!("...so that {} ({})",
                                    desc, values_str)[]);
                    }
                    None => {
                        // Really should avoid printing this error at
                        // all, since it is derived, but that would
                        // require more refactoring than I feel like
                        // doing right now. - nmatsakis
                        self.tcx.sess.span_note(
                            trace.origin.span(),
                            &format!("...so that {}", desc)[]);
                    }
                }
            }
            infer::Reborrow(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that reference does not outlive \
                    borrowed content");
            }
            infer::ReborrowUpvar(span, ref upvar_id) => {
                self.tcx.sess.span_note(
                    span,
                    &format!(
                        "...so that closure can access `{}`",
                        ty::local_var_name_str(self.tcx, upvar_id.var_id)
                            .to_string())[])
            }
            infer::InfStackClosure(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that closure does not outlive its stack frame");
            }
            infer::InvokeClosure(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that closure is not invoked outside its lifetime");
            }
            infer::DerefPointer(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that pointer is not dereferenced \
                    outside its lifetime");
            }
            infer::FreeVariable(span, id) => {
                self.tcx.sess.span_note(
                    span,
                    &format!("...so that captured variable `{}` \
                            does not outlive the enclosing closure",
                            ty::local_var_name_str(
                                self.tcx,
                                id).to_string())[]);
            }
            infer::IndexSlice(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that slice is not indexed outside the lifetime");
            }
            infer::RelateObjectBound(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that it can be closed over into an object");
            }
            infer::CallRcvr(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that method receiver is valid for the method call");
            }
            infer::CallArg(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that argument is valid for the call");
            }
            infer::CallReturn(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that return value is valid for the call");
            }
            infer::Operand(span) => {
                self.tcx.sess.span_err(
                    span,
                    "...so that operand is valid for operation");
            }
            infer::AddrOf(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that reference is valid \
                     at the time of borrow");
            }
            infer::AutoBorrow(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that auto-reference is valid \
                     at the time of borrow");
            }
            infer::ExprTypeIsNotInScope(t, span) => {
                self.tcx.sess.span_note(
                    span,
                    &format!("...so type `{}` of expression is valid during the \
                             expression",
                            self.ty_to_string(t))[]);
            }
            infer::BindingTypeIsNotValidAtDecl(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that variable is valid at time of its declaration");
            }
            infer::ReferenceOutlivesReferent(ty, span) => {
                self.tcx.sess.span_note(
                    span,
                    &format!("...so that the reference type `{}` \
                             does not outlive the data it points at",
                            self.ty_to_string(ty))[]);
            }
            infer::RelateParamBound(span, t) => {
                self.tcx.sess.span_note(
                    span,
                    &format!("...so that the type `{}` \
                             will meet its required lifetime bounds",
                            self.ty_to_string(t))[]);
            }
            infer::RelateDefaultParamBound(span, t) => {
                self.tcx.sess.span_note(
                    span,
                    &format!("...so that type parameter \
                             instantiated with `{}`, \
                             will meet its declared lifetime bounds",
                            self.ty_to_string(t))[]);
            }
            infer::RelateRegionParamBound(span) => {
                self.tcx.sess.span_note(
                    span,
                    &format!("...so that the declared lifetime parameter bounds \
                                are satisfied")[]);
            }
            infer::SafeDestructor(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that references are valid when the destructor \
                     runs")
            }
        }
    }
}

pub trait Resolvable<'tcx> {
    fn resolve<'a>(&self, infcx: &InferCtxt<'a, 'tcx>) -> Self;
    fn contains_error(&self) -> bool;
}

impl<'tcx> Resolvable<'tcx> for Ty<'tcx> {
    fn resolve<'a>(&self, infcx: &InferCtxt<'a, 'tcx>) -> Ty<'tcx> {
        infcx.resolve_type_vars_if_possible(self)
    }
    fn contains_error(&self) -> bool {
        ty::type_is_error(*self)
    }
}

impl<'tcx> Resolvable<'tcx> for Rc<ty::TraitRef<'tcx>> {
    fn resolve<'a>(&self, infcx: &InferCtxt<'a, 'tcx>)
                   -> Rc<ty::TraitRef<'tcx>> {
        Rc::new(infcx.resolve_type_vars_if_possible(&**self))
    }
    fn contains_error(&self) -> bool {
        ty::trait_ref_contains_error(&**self)
    }
}

impl<'tcx> Resolvable<'tcx> for ty::PolyTraitRef<'tcx> {
    fn resolve<'a>(&self,
                   infcx: &InferCtxt<'a, 'tcx>)
                   -> ty::PolyTraitRef<'tcx>
    {
        infcx.resolve_type_vars_if_possible(self)
    }

    fn contains_error(&self) -> bool {
        ty::trait_ref_contains_error(&*self.0)
    }
}

fn lifetimes_in_scope(tcx: &ty::ctxt,
                      scope_id: ast::NodeId)
                      -> Vec<ast::LifetimeDef> {
    let mut taken = Vec::new();
    let parent = tcx.map.get_parent(scope_id);
    let method_id_opt = match tcx.map.find(parent) {
        Some(node) => match node {
            ast_map::NodeItem(item) => match item.node {
                ast::ItemFn(_, _, _, ref gen, _) => {
                    taken.push_all(&gen.lifetimes[]);
                    None
                },
                _ => None
            },
            ast_map::NodeImplItem(ii) => {
                match *ii {
                    ast::MethodImplItem(ref m) => {
                        taken.push_all(&m.pe_generics().lifetimes[]);
                        Some(m.id)
                    }
                    ast::TypeImplItem(_) => None,
                }
            }
            _ => None
        },
        None => None
    };
    if method_id_opt.is_some() {
        let method_id = method_id_opt.unwrap();
        let parent = tcx.map.get_parent(method_id);
        match tcx.map.find(parent) {
            Some(node) => match node {
                ast_map::NodeItem(item) => match item.node {
                    ast::ItemImpl(_, _, ref gen, _, _, _) => {
                        taken.push_all(&gen.lifetimes);
                    }
                    _ => ()
                },
                _ => ()
            },
            None => ()
        }
    }
    return taken;
}

// LifeGiver is responsible for generating fresh lifetime names
struct LifeGiver {
    taken: HashSet<String>,
    counter: Cell<uint>,
    generated: RefCell<Vec<ast::Lifetime>>,
}

impl LifeGiver {
    fn with_taken(taken: &[ast::LifetimeDef]) -> LifeGiver {
        let mut taken_ = HashSet::new();
        for lt in taken {
            let lt_name = token::get_name(lt.lifetime.name).to_string();
            taken_.insert(lt_name);
        }
        LifeGiver {
            taken: taken_,
            counter: Cell::new(0),
            generated: RefCell::new(Vec::new()),
        }
    }

    fn inc_counter(&self) {
        let c = self.counter.get();
        self.counter.set(c+1);
    }

    fn give_lifetime(&self) -> ast::Lifetime {
        let mut lifetime;
        loop {
            let mut s = String::from_str("'");
            s.push_str(&num_to_string(self.counter.get())[]);
            if !self.taken.contains(&s) {
                lifetime = name_to_dummy_lifetime(
                                    token::str_to_ident(&s[..]).name);
                self.generated.borrow_mut().push(lifetime);
                break;
            }
            self.inc_counter();
        }
        self.inc_counter();
        return lifetime;

        // 0 .. 25 generates a .. z, 26 .. 51 generates aa .. zz, and so on
        fn num_to_string(counter: uint) -> String {
            let mut s = String::new();
            let (n, r) = (counter/26 + 1, counter % 26);
            let letter: char = from_u32((r+97) as u32).unwrap();
            for _ in 0..n {
                s.push(letter);
            }
            s
        }
    }

    fn get_generated_lifetimes(&self) -> Vec<ast::Lifetime> {
        self.generated.borrow().clone()
    }
}
