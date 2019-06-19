use crate::infer::{self, InferCtxt, SubregionOrigin};
use crate::middle::region;
use crate::ty::{self, Region};
use crate::ty::error::TypeError;
use errors::DiagnosticBuilder;

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    pub(super) fn note_region_origin(&self,
                                     err: &mut DiagnosticBuilder<'_>,
                                     origin: &SubregionOrigin<'tcx>) {
        match *origin {
            infer::Subtype(ref trace) => {
                if let Some((expected, found)) = self.values_str(&trace.values) {
                    let expected = expected.content();
                    let found = found.content();
                    err.note(&format!("...so that the {}:\nexpected {}\n   found {}",
                                      trace.cause.as_requirement_str(),
                                      expected,
                                      found));
                } else {
                    // FIXME: this really should be handled at some earlier stage. Our
                    // handling of region checking when type errors are present is
                    // *terrible*.

                    err.span_note(trace.cause.span,
                                  &format!("...so that {}", trace.cause.as_requirement_str()));
                }
            }
            infer::Reborrow(span) => {
                err.span_note(span,
                              "...so that reference does not outlive borrowed content");
            }
            infer::ReborrowUpvar(span, ref upvar_id) => {
                let var_name = self.tcx.hir().name(upvar_id.var_path.hir_id);
                err.span_note(span,
                              &format!("...so that closure can access `{}`", var_name));
            }
            infer::InfStackClosure(span) => {
                err.span_note(span, "...so that closure does not outlive its stack frame");
            }
            infer::InvokeClosure(span) => {
                err.span_note(span,
                              "...so that closure is not invoked outside its lifetime");
            }
            infer::DerefPointer(span) => {
                err.span_note(span,
                              "...so that pointer is not dereferenced outside its lifetime");
            }
            infer::ClosureCapture(span, id) => {
                err.span_note(span,
                              &format!("...so that captured variable `{}` does not outlive the \
                                        enclosing closure",
                                       self.tcx.hir().name(id)));
            }
            infer::IndexSlice(span) => {
                err.span_note(span, "...so that slice is not indexed outside the lifetime");
            }
            infer::RelateObjectBound(span) => {
                err.span_note(span, "...so that it can be closed over into an object");
            }
            infer::CallRcvr(span) => {
                err.span_note(span,
                              "...so that method receiver is valid for the method call");
            }
            infer::CallArg(span) => {
                err.span_note(span, "...so that argument is valid for the call");
            }
            infer::CallReturn(span) => {
                err.span_note(span, "...so that return value is valid for the call");
            }
            infer::Operand(span) => {
                err.span_note(span, "...so that operand is valid for operation");
            }
            infer::AddrOf(span) => {
                err.span_note(span, "...so that reference is valid at the time of borrow");
            }
            infer::AutoBorrow(span) => {
                err.span_note(span,
                              "...so that auto-reference is valid at the time of borrow");
            }
            infer::ExprTypeIsNotInScope(t, span) => {
                err.span_note(span,
                              &format!("...so type `{}` of expression is valid during the \
                                        expression",
                                       self.ty_to_string(t)));
            }
            infer::BindingTypeIsNotValidAtDecl(span) => {
                err.span_note(span,
                              "...so that variable is valid at time of its declaration");
            }
            infer::ParameterInScope(_, span) => {
                err.span_note(span,
                              "...so that a type/lifetime parameter is in scope here");
            }
            infer::DataBorrowed(ty, span) => {
                err.span_note(span,
                              &format!("...so that the type `{}` is not borrowed for too long",
                                       self.ty_to_string(ty)));
            }
            infer::ReferenceOutlivesReferent(ty, span) => {
                err.span_note(span,
                              &format!("...so that the reference type `{}` does not outlive the \
                                        data it points at",
                                       self.ty_to_string(ty)));
            }
            infer::RelateParamBound(span, t) => {
                err.span_note(span,
                              &format!("...so that the type `{}` will meet its required \
                                        lifetime bounds",
                                       self.ty_to_string(t)));
            }
            infer::RelateDefaultParamBound(span, t) => {
                err.span_note(span,
                              &format!("...so that type parameter instantiated with `{}`, will \
                                        meet its declared lifetime bounds",
                                       self.ty_to_string(t)));
            }
            infer::RelateRegionParamBound(span) => {
                err.span_note(span,
                              "...so that the declared lifetime parameter bounds are satisfied");
            }
            infer::SafeDestructor(span) => {
                err.span_note(span,
                              "...so that references are valid when the destructor runs");
            }
            infer::CompareImplMethodObligation { span, .. } => {
                err.span_note(span,
                              "...so that the definition in impl matches the definition from the \
                               trait");
            }
        }
    }

    pub(super) fn report_concrete_failure(&self,
                                          region_scope_tree: &region::ScopeTree,
                                          origin: SubregionOrigin<'tcx>,
                                          sub: Region<'tcx>,
                                          sup: Region<'tcx>)
                                          -> DiagnosticBuilder<'tcx> {
        match origin {
            infer::Subtype(trace) => {
                let terr = TypeError::RegionsDoesNotOutlive(sup, sub);
                let mut err = self.report_and_explain_type_error(trace, &terr);
                self.tcx.note_and_explain_region(region_scope_tree, &mut err, "", sup, "...");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "...does not necessarily outlive ", sub, "");
                err
            }
            infer::Reborrow(span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0312,
                                               "lifetime of reference outlives lifetime of \
                                                borrowed content...");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                                                 "...the reference is valid for ",
                                                 sub,
                                                 "...");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                                                 "...but the borrowed content is only valid for ",
                                                 sup,
                                                 "");
                err
            }
            infer::ReborrowUpvar(span, ref upvar_id) => {
                let var_name = self.tcx.hir().name(upvar_id.var_path.hir_id);
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0313,
                                               "lifetime of borrowed pointer outlives lifetime \
                                                of captured variable `{}`...",
                                               var_name);
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                                                 "...the borrowed pointer is valid for ",
                                                 sub,
                                                 "...");
                self.tcx.note_and_explain_region(
                    region_scope_tree,
                    &mut err,
                    &format!("...but `{}` is only valid for ", var_name),
                    sup,
                    "");
                err
            }
            infer::InfStackClosure(span) => {
                let mut err =
                    struct_span_err!(self.tcx.sess, span, E0314, "closure outlives stack frame");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                                                 "...the closure must be valid for ",
                                                 sub,
                                                 "...");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                                                 "...but the closure's stack frame is only valid \
                                                  for ",
                                                 sup,
                                                 "");
                err
            }
            infer::InvokeClosure(span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0315,
                                               "cannot invoke closure outside of its lifetime");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "the closure is only valid for ", sup, "");
                err
            }
            infer::DerefPointer(span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0473,
                                               "dereference of reference outside its lifetime");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "the reference is only valid for ", sup, "");
                err
            }
            infer::ClosureCapture(span, id) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0474,
                                               "captured variable `{}` does not outlive the \
                                                enclosing closure",
                                               self.tcx.hir().name(id));
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "captured variable is valid for ", sup, "");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "closure is valid for ", sub, "");
                err
            }
            infer::IndexSlice(span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0475,
                                               "index of slice outside its lifetime");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "the slice is only valid for ", sup, "");
                err
            }
            infer::RelateObjectBound(span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0476,
                                               "lifetime of the source pointer does not outlive \
                                                lifetime bound of the object type");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "object type is valid for ", sub, "");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                                                 "source pointer is only valid for ",
                                                 sup,
                                                 "");
                err
            }
            infer::RelateParamBound(span, ty) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0477,
                                               "the type `{}` does not fulfill the required \
                                                lifetime",
                                               self.ty_to_string(ty));
                match *sub {
                    ty::ReStatic => {
                        self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                            "type must satisfy ", sub, "")
                    }
                    _ => {
                        self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                            "type must outlive ", sub, "")
                    }
                }
                err
            }
            infer::RelateRegionParamBound(span) => {
                let mut err =
                    struct_span_err!(self.tcx.sess, span, E0478, "lifetime bound not satisfied");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                                                 "lifetime parameter instantiated with ",
                                                 sup,
                                                 "");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                                                 "but lifetime parameter must outlive ",
                                                 sub,
                                                 "");
                err
            }
            infer::RelateDefaultParamBound(span, ty) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0479,
                                               "the type `{}` (provided as the value of a type \
                                                parameter) is not valid at this point",
                                               self.ty_to_string(ty));
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "type must outlive ", sub, "");
                err
            }
            infer::CallRcvr(span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0480,
                                               "lifetime of method receiver does not outlive the \
                                                method call");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                "the receiver is only valid for ", sup, "");
                err
            }
            infer::CallArg(span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0481,
                                               "lifetime of function argument does not outlive \
                                                the function call");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                                                 "the function argument is only valid for ",
                                                 sup,
                                                 "");
                err
            }
            infer::CallReturn(span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0482,
                                               "lifetime of return value does not outlive the \
                                                function call");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                                                 "the return value is only valid for ",
                                                 sup,
                                                 "");
                err
            }
            infer::Operand(span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0483,
                                               "lifetime of operand does not outlive the \
                                                operation");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "the operand is only valid for ", sup, "");
                err
            }
            infer::AddrOf(span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0484,
                                               "reference is not valid at the time of borrow");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "the borrow is only valid for ", sup, "");
                err
            }
            infer::AutoBorrow(span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0485,
                                               "automatically reference is not valid at the time \
                                                of borrow");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                                                 "the automatic borrow is only valid for ",
                                                 sup,
                                                 "");
                err
            }
            infer::ExprTypeIsNotInScope(t, span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0486,
                                               "type of expression contains references that are \
                                                not valid during the expression: `{}`",
                                               self.ty_to_string(t));
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "type is only valid for ", sup, "");
                err
            }
            infer::SafeDestructor(span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0487,
                                               "unsafe use of destructor: destructor might be \
                                                called while references are dead");
                // FIXME (22171): terms "super/subregion" are suboptimal
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "superregion: ", sup, "");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "subregion: ", sub, "");
                err
            }
            infer::BindingTypeIsNotValidAtDecl(span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0488,
                                               "lifetime of variable does not enclose its \
                                                declaration");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "the variable is only valid for ", sup, "");
                err
            }
            infer::ParameterInScope(_, span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0489,
                                               "type/lifetime parameter not in scope here");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "the parameter is only valid for ", sub, "");
                err
            }
            infer::DataBorrowed(ty, span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0490,
                                               "a value of type `{}` is borrowed for too long",
                                               self.ty_to_string(ty));
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "the type is valid for ", sub, "");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "but the borrow lasts for ", sup, "");
                err
            }
            infer::ReferenceOutlivesReferent(ty, span) => {
                let mut err = struct_span_err!(self.tcx.sess,
                                               span,
                                               E0491,
                                               "in type `{}`, reference has a longer lifetime \
                                                than the data it references",
                                               self.ty_to_string(ty));
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                    "the pointer is valid for ", sub, "");
                self.tcx.note_and_explain_region(region_scope_tree, &mut err,
                                                 "but the referenced data is only valid for ",
                                                 sup,
                                                 "");
                err
            }
            infer::CompareImplMethodObligation { span,
                                                 item_name,
                                                 impl_item_def_id,
                                                 trait_item_def_id } => {
                self.report_extra_impl_obligation(span,
                                                  item_name,
                                                  impl_item_def_id,
                                                  trait_item_def_id,
                                                  &format!("`{}: {}`", sup, sub))
            }
        }
    }

    pub(super) fn report_placeholder_failure(
        &self,
        region_scope_tree: &region::ScopeTree,
        placeholder_origin: SubregionOrigin<'tcx>,
        sub: Region<'tcx>,
        sup: Region<'tcx>,
    ) -> DiagnosticBuilder<'tcx> {
        // I can't think how to do better than this right now. -nikomatsakis
        match placeholder_origin {
            infer::Subtype(trace) => {
                let terr = TypeError::RegionsPlaceholderMismatch;
                self.report_and_explain_type_error(trace, &terr)
            }

            _ => {
                self.report_concrete_failure(region_scope_tree, placeholder_origin, sub, sup)
            }
        }
    }
}
