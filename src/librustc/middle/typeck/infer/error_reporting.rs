// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Error Reporting Code for the inference engine

Because of the way inference, and in particular region inference,
works, it often happens that errors are not detected until far after
the relevant line of code has been type-checked. Therefore, there is
an elaborate system to track why a particular constraint in the
inference graph arose so that we can explain to the user what gave
rise to a patricular error.

The basis of the system are the "origin" types. An "origin" is the
reason that a constraint or inference variable arose. There are
different "origin" enums for different kinds of constraints/variables
(e.g., `TypeOrigin`, `RegionVariableOrigin`). An origin always has
a span, but also more information so that we can generate a meaningful
error message.

Having a catalogue of all the different reasons an error can arise is
also useful for other reasons, like cross-referencing FAQs etc, though
we are not really taking advantage of this yet.

# Region Inference

Region inference is particularly tricky because it always succeeds "in
the moment" and simply registers a constraint. Then, at the end, we
can compute the full graph and report errors, so we need to be able to
store and later report what gave rise to the conflicting constraints.

# Subtype Trace

Determing whether `T1 <: T2` often involves a number of subtypes and
subconstraints along the way. A "TypeTrace" is an extended version
of an origin that traces the types and other values that were being
compared. It is not necessarily comprehensive (in fact, at the time of
this writing it only tracks the root values being compared) but I'd
like to extend it to include significant "waypoints". For example, if
you are comparing `(T1, T2) <: (T3, T4)`, and the problem is that `T2
<: T4` fails, I'd like the trace to include enough information to say
"in the 2nd element of the tuple". Similarly, failures when comparing
arguments or return types in fn types should be able to cite the
specific position, etc.

# Reality vs plan

Of course, there is still a LOT of code in typeck that has yet to be
ported to this system, and which relies on string concatenation at the
time of error detection.

*/

use middle::ty;
use middle::ty::Region;
use middle::typeck::infer;
use middle::typeck::infer::InferCtxt;
use middle::typeck::infer::TypeTrace;
use middle::typeck::infer::SubregionOrigin;
use middle::typeck::infer::RegionVariableOrigin;
use middle::typeck::infer::ValuePairs;
use middle::typeck::infer::region_inference::RegionResolutionError;
use middle::typeck::infer::region_inference::ConcreteFailure;
use middle::typeck::infer::region_inference::SubSupConflict;
use middle::typeck::infer::region_inference::SupSupConflict;
use syntax::opt_vec::OptVec;
use util::ppaux::UserString;
use util::ppaux::bound_region_to_str;
use util::ppaux::note_and_explain_region;

pub trait ErrorReporting {
    fn report_region_errors(@mut self,
                            errors: &OptVec<RegionResolutionError>);

    fn report_and_explain_type_error(@mut self,
                                     trace: TypeTrace,
                                     terr: &ty::type_err);

    fn values_str(@mut self, values: &ValuePairs) -> Option<~str>;

    fn expected_found_str<T:UserString+Resolvable>(
        @mut self,
        exp_found: &ty::expected_found<T>)
        -> Option<~str>;

    fn report_concrete_failure(@mut self,
                               origin: SubregionOrigin,
                               sub: Region,
                               sup: Region);

    fn report_sub_sup_conflict(@mut self,
                               var_origin: RegionVariableOrigin,
                               sub_origin: SubregionOrigin,
                               sub_region: Region,
                               sup_origin: SubregionOrigin,
                               sup_region: Region);

    fn report_sup_sup_conflict(@mut self,
                               var_origin: RegionVariableOrigin,
                               origin1: SubregionOrigin,
                               region1: Region,
                               origin2: SubregionOrigin,
                               region2: Region);
}

trait ErrorReportingHelpers {
    fn report_inference_failure(@mut self,
                                var_origin: RegionVariableOrigin);

    fn note_region_origin(@mut self,
                          origin: SubregionOrigin);
}

impl ErrorReporting for InferCtxt {
    fn report_region_errors(@mut self,
                            errors: &OptVec<RegionResolutionError>) {
        for error in errors.iter() {
            match *error {
                ConcreteFailure(origin, sub, sup) => {
                    self.report_concrete_failure(origin, sub, sup);
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
            }
        }
    }

    fn report_and_explain_type_error(@mut self,
                                     trace: TypeTrace,
                                     terr: &ty::type_err) {
        let tcx = self.tcx;

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
            infer::MatchExpression(_) => "match arms have incompatible types",
            infer::IfExpression(_) => "if and else have incompatible types",
        };

        self.tcx.sess.span_err(
            trace.origin.span(),
            format!("{}: {} ({})",
                 message_root_str,
                 expected_found_str,
                 ty::type_err_to_str(tcx, terr)));

        ty::note_and_explain_type_err(self.tcx, terr);
    }

    fn values_str(@mut self, values: &ValuePairs) -> Option<~str> {
        /*!
         * Returns a string of the form "expected `{}` but found `{}`",
         * or None if this is a derived error.
         */
        match *values {
            infer::Types(ref exp_found) => {
                self.expected_found_str(exp_found)
            }
            infer::TraitRefs(ref exp_found) => {
                self.expected_found_str(exp_found)
            }
        }
    }

    fn expected_found_str<T:UserString+Resolvable>(
        @mut self,
        exp_found: &ty::expected_found<T>)
        -> Option<~str>
    {
        let expected = exp_found.expected.resolve(self);
        if expected.contains_error() {
            return None;
        }

        let found = exp_found.found.resolve(self);
        if found.contains_error() {
            return None;
        }

        Some(format!("expected `{}` but found `{}`",
                  expected.user_string(self.tcx),
                  found.user_string(self.tcx)))
    }

    fn report_concrete_failure(@mut self,
                               origin: SubregionOrigin,
                               sub: Region,
                               sup: Region) {
        match origin {
            infer::Subtype(trace) => {
                let terr = ty::terr_regions_does_not_outlive(sup, sub);
                self.report_and_explain_type_error(trace, &terr);
            }
            infer::Reborrow(span) => {
                self.tcx.sess.span_err(
                    span,
                    "lifetime of borrowed pointer outlines \
                     lifetime of borrowed content...");
                note_and_explain_region(
                    self.tcx,
                    "...the borrowed pointer is valid for ",
                    sub,
                    "...");
                note_and_explain_region(
                    self.tcx,
                    "...but the borrowed content is only valid for ",
                    sup,
                    "");
            }
            infer::InfStackClosure(span) => {
                self.tcx.sess.span_err(
                    span,
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
                self.tcx.sess.span_err(
                    span,
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
            infer::FreeVariable(span) => {
                self.tcx.sess.span_err(
                    span,
                    "captured variable does not outlive the enclosing closure");
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
                self.tcx.sess.span_err(
                    span,
                    format!("index of slice outside its lifetime"));
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
            infer::AddrOf(span) => {
                self.tcx.sess.span_err(
                    span,
                    "borrowed pointer is not valid \
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
                    "automatically borrowed pointer is not valid \
                     at the time of borrow");
                note_and_explain_region(
                    self.tcx,
                    "the automatic borrow is only valid for ",
                    sup,
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
                    format!("in type `{}`, pointer has a longer lifetime than \
                          the data it references",
                         ty.user_string(self.tcx)));
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

    fn report_sub_sup_conflict(@mut self,
                               var_origin: RegionVariableOrigin,
                               sub_origin: SubregionOrigin,
                               sub_region: Region,
                               sup_origin: SubregionOrigin,
                               sup_region: Region) {
        self.report_inference_failure(var_origin);

        note_and_explain_region(
            self.tcx,
            "first, the lifetime cannot outlive ",
            sup_region,
            "...");

        self.note_region_origin(sup_origin);

        note_and_explain_region(
            self.tcx,
            "but, the lifetime must be valid for ",
            sub_region,
            "...");

        self.note_region_origin(sub_origin);
    }

    fn report_sup_sup_conflict(@mut self,
                               var_origin: RegionVariableOrigin,
                               origin1: SubregionOrigin,
                               region1: Region,
                               origin2: SubregionOrigin,
                               region2: Region) {
        self.report_inference_failure(var_origin);

        note_and_explain_region(
            self.tcx,
            "first, the lifetime must be contained by ",
            region1,
            "...");

        self.note_region_origin(origin1);

        note_and_explain_region(
            self.tcx,
            "but, the lifetime must also be contained by ",
            region2,
            "...");

        self.note_region_origin(origin2);
    }
}

impl ErrorReportingHelpers for InferCtxt {
    fn report_inference_failure(@mut self,
                                var_origin: RegionVariableOrigin) {
        let var_description = match var_origin {
            infer::MiscVariable(_) => ~"",
            infer::PatternRegion(_) => ~" for pattern",
            infer::AddrOfRegion(_) => ~" for borrow expression",
            infer::AddrOfSlice(_) => ~" for slice expression",
            infer::Autoref(_) => ~" for autoref",
            infer::Coercion(_) => ~" for automatic coercion",
            infer::BoundRegionInFnCall(_, br) => {
                format!(" for {}in function call",
                        bound_region_to_str(self.tcx, "region ", true, br))
            }
            infer::BoundRegionInFnType(_, br) => {
                format!(" for {}in function type",
                        bound_region_to_str(self.tcx, "region ", true, br))
            }
            infer::BoundRegionInTypeOrImpl(_) => {
                format!(" for region in type/impl")
            }
            infer::BoundRegionInCoherence(*) => {
                format!(" for coherence check")
            }
        };

        self.tcx.sess.span_err(
            var_origin.span(),
            format!("cannot infer an appropriate lifetime{} \
                    due to conflicting requirements",
                    var_description));
    }

    fn note_region_origin(@mut self,
                          origin: SubregionOrigin) {
        match origin {
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
                        format!("type matches impl")
                    }
                    infer::MatchExpression(_) => {
                        format!("match arms have compatible types")
                    }
                    infer::IfExpression(_) => {
                        format!("if and else have compatible types")
                    }
                };

                match self.values_str(&trace.values) {
                    Some(values_str) => {
                        self.tcx.sess.span_note(
                            trace.origin.span(),
                            format!("...so that {} ({})",
                                    desc, values_str));
                    }
                    None => {
                        // Really should avoid printing this error at
                        // all, since it is derived, but that would
                        // require more refactoring than I feel like
                        // doing right now. - nmatsakis
                        self.tcx.sess.span_note(
                            trace.origin.span(),
                            format!("...so that {}", desc));
                    }
                }
            }
            infer::Reborrow(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that borrowed pointer does not outlive \
                    borrowed content");
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
            infer::FreeVariable(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that captured variable does not outlive the \
                    enclosing closure");
            }
            infer::IndexSlice(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that slice is not indexed outside the lifetime");
            }
            infer::RelateObjectBound(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that source pointer does not outlive \
                     lifetime bound of the object type");
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
            infer::AddrOf(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that borrowed pointer is valid \
                     at the time of borrow");
            }
            infer::AutoBorrow(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that automatically borrowed pointer is valid \
                     at the time of borrow");
            }
            infer::BindingTypeIsNotValidAtDecl(span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that variable is valid at time of its declaration");
            }
            infer::ReferenceOutlivesReferent(_, span) => {
                self.tcx.sess.span_note(
                    span,
                    "...so that the pointer does not outlive the \
                    data it points at");
            }
        }
    }
}

trait Resolvable {
    fn resolve(&self, infcx: @mut InferCtxt) -> Self;
    fn contains_error(&self) -> bool;
}

impl Resolvable for ty::t {
    fn resolve(&self, infcx: @mut InferCtxt) -> ty::t {
        infcx.resolve_type_vars_if_possible(*self)
    }
    fn contains_error(&self) -> bool {
        ty::type_is_error(*self)
    }
}

impl Resolvable for @ty::TraitRef {
    fn resolve(&self, infcx: @mut InferCtxt) -> @ty::TraitRef {
        @infcx.resolve_type_vars_in_trait_ref_if_possible(*self)
    }
    fn contains_error(&self) -> bool {
        ty::trait_ref_contains_error(*self)
    }
}

