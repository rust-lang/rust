#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

use rustc_errors::{
    struct_span_err, DiagnosticBuilder, DiagnosticId, DiagnosticMessage, ErrorGuaranteed, MultiSpan,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;

use crate::session_diagnostics::{
    ActMovedValueErr, AssignBorrowErr, AssignErr, BorrowAcrossDestructor,
    BorrowAcrossGeneratorYield, BorrowEscapeClosure, ClosureConstructLabel,
    ClosureUniquelyBorrowErr, ClosureVarOutliveErr, ImmuteArgAssign, ImmuteVarReassign,
    InteriorDropMoveErr, InteriorNoncopyMoveErr, MoveBorrowedErr, MovedOutErr, MutateInImmute,
    PathShortLive, ReturnRefLocalErr, TemporaryDroppedErr, ThreadLocalOutliveErr,
    TwoClosuresUniquelyBorrowErr, UniquelyBorrowReborrowErr, UseMutBorrowErr,
};

impl<'cx, 'tcx> crate::MirBorrowckCtxt<'cx, 'tcx> {
    pub(crate) fn cannot_move_when_borrowed(
        &self,
        span: Span,
        desc: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(MoveBorrowedErr { desc, span })
    }

    pub(crate) fn cannot_use_when_mutably_borrowed(
        &self,
        span: Span,
        desc: &str,
        borrow_span: Span,
        borrow_desc: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(UseMutBorrowErr { desc, borrow_desc, span, borrow_span })
    }

    pub(crate) fn cannot_mutably_borrow_multiply(
        &self,
        new_loan_span: Span,
        desc: &str,
        opt_via: &str,
        old_loan_span: Span,
        old_opt_via: &str,
        old_load_end_span: Option<Span>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        //FIXME: migrate later
        let via =
            |msg: &str| if msg.is_empty() { "".to_string() } else { format!(" (via {})", msg) };
        let mut err = struct_span_err!(
            self,
            new_loan_span,
            E0499,
            "cannot borrow {}{} as mutable more than once at a time",
            desc,
            via(opt_via),
        );
        if old_loan_span == new_loan_span {
            // Both borrows are happening in the same place
            // Meaning the borrow is occurring in a loop
            err.span_label(
                new_loan_span,
                format!(
                    "{}{} was mutably borrowed here in the previous iteration of the loop{}",
                    desc,
                    via(opt_via),
                    opt_via,
                ),
            );
            if let Some(old_load_end_span) = old_load_end_span {
                err.span_label(old_load_end_span, "mutable borrow ends here");
            }
        } else {
            err.span_label(
                old_loan_span,
                format!("first mutable borrow occurs here{}", via(old_opt_via)),
            );
            err.span_label(
                new_loan_span,
                format!("second mutable borrow occurs here{}", via(opt_via)),
            );
            if let Some(old_load_end_span) = old_load_end_span {
                err.span_label(old_load_end_span, "first borrow ends here");
            }
        }
        err
    }

    pub(crate) fn cannot_uniquely_borrow_by_two_closures(
        &self,
        new_loan_span: Span,
        desc: &str,
        old_loan_span: Span,
        old_load_end_span: Option<Span>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        let case: ClosureConstructLabel;
        let diff_span: Option<Span>;
        if old_loan_span == new_loan_span {
            case = ClosureConstructLabel::Both { old_loan_span };
            diff_span = None;
        } else {
            case = ClosureConstructLabel::First { old_loan_span };
            diff_span = Some(new_loan_span);
        }
        self.infcx.tcx.sess.create_err(TwoClosuresUniquelyBorrowErr {
            desc,
            case,
            new_loan_span,
            old_load_end_span,
            diff_span,
        })
    }

    pub(crate) fn cannot_uniquely_borrow_by_one_closure(
        &self,
        new_loan_span: Span,
        container_name: &str,
        desc_new: &str,
        opt_via: &str,
        old_loan_span: Span,
        noun_old: &str,
        old_opt_via: &str,
        previous_end_span: Option<Span>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(ClosureUniquelyBorrowErr {
            container_name,
            desc_new,
            opt_via,
            noun_old,
            old_opt_via,
            new_loan_span,
            old_loan_span,
            previous_end_span,
        })
    }

    pub(crate) fn cannot_reborrow_already_uniquely_borrowed(
        &self,
        new_loan_span: Span,
        container_name: &str,
        desc_new: &str,
        opt_via: &str,
        kind_new: &str,
        old_loan_span: Span,
        old_opt_via: &str,
        previous_end_span: Option<Span>,
        second_borrow_desc: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(UniquelyBorrowReborrowErr {
            container_name,
            desc_new,
            opt_via,
            kind_new,
            old_opt_via,
            second_borrow_desc,
            new_loan_span,
            old_loan_span,
            previous_end_span,
        })
    }

    pub(crate) fn cannot_reborrow_already_borrowed(
        &self,
        span: Span,
        desc_new: &str,
        msg_new: &str,
        kind_new: &str,
        old_span: Span,
        noun_old: &str,
        kind_old: &str,
        msg_old: &str,
        old_load_end_span: Option<Span>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        //FIXME: would it be better for manual impl for this case?
        let via =
            |msg: &str| if msg.is_empty() { "".to_string() } else { format!(" (via {})", msg) };
        let mut err = struct_span_err!(
            self,
            span,
            E0502,
            "cannot borrow {}{} as {} because {} is also borrowed as {}{}",
            desc_new,
            via(msg_new),
            kind_new,
            noun_old,
            kind_old,
            via(msg_old),
        );

        if msg_new == "" {
            // If `msg_new` is empty, then this isn't a borrow of a union field.
            err.span_label(span, format!("{} borrow occurs here", kind_new));
            err.span_label(old_span, format!("{} borrow occurs here", kind_old));
        } else {
            // If `msg_new` isn't empty, then this a borrow of a union field.
            err.span_label(
                span,
                format!(
                    "{} borrow of {} -- which overlaps with {} -- occurs here",
                    kind_new, msg_new, msg_old,
                ),
            );
            err.span_label(old_span, format!("{} borrow occurs here{}", kind_old, via(msg_old)));
        }

        if let Some(old_load_end_span) = old_load_end_span {
            err.span_label(old_load_end_span, format!("{} borrow ends here", kind_old));
        }
        err
    }

    pub(crate) fn cannot_assign_to_borrowed(
        &self,
        span: Span,
        borrow_span: Span,
        desc: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(AssignBorrowErr { desc, span, borrow_span })
    }

    pub(crate) fn cannot_reassign_immutable(
        &self,
        span: Span,
        desc: &str,
        is_arg: bool,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        if is_arg {
            self.infcx.tcx.sess.create_err(ImmuteArgAssign { desc, span })
        } else {
            self.infcx.tcx.sess.create_err(ImmuteVarReassign { desc, span })
        }
    }

    pub(crate) fn cannot_assign(
        &self,
        span: Span,
        desc: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(AssignErr { desc, span })
    }

    pub(crate) fn cannot_move_out_of(
        &self,
        move_from_span: Span,
        move_from_desc: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(MovedOutErr { move_from_desc, move_from_span })
    }

    /// Signal an error due to an attempt to move out of the interior
    /// of an array or slice. `is_index` is None when error origin
    /// didn't capture whether there was an indexing operation or not.
    pub(crate) fn cannot_move_out_of_interior_noncopy(
        &self,
        move_from_span: Span,
        ty: Ty<'_>,
        is_index: Option<bool>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        let type_name = match (&ty.kind(), is_index) {
            (&ty::Array(_, _), Some(true)) | (&ty::Array(_, _), None) => "array",
            (&ty::Slice(_), _) => "slice",
            _ => span_bug!(move_from_span, "this path should not cause illegal move"),
        };
        //FIXME: failed ui test diag-migration
        //       -    error[E0508]: cannot move out of type `[S; 1]`, a non-copy array
        //       +    error[E0508]: cannot move out of type `S`, a non-copy array
        self.infcx.tcx.sess.create_err(InteriorNoncopyMoveErr { ty, type_name, move_from_span })
    }

    pub(crate) fn cannot_move_out_of_interior_of_drop(
        &self,
        move_from_span: Span,
        container_ty: Ty<'_>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(InteriorDropMoveErr { container_ty, move_from_span })
    }

    pub(crate) fn cannot_act_on_moved_value(
        &self,
        use_span: Span,
        verb: &str,
        optional_adverb_for_moved: &str,
        moved_path: Option<String>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let moved_path = moved_path.map(|mp| format!(": `{}`", mp)).unwrap_or_default();
        self.infcx.tcx.sess.create_err(ActMovedValueErr {
            verb,
            optional_adverb_for_moved,
            moved_path,
            use_span,
        })
    }

    //FIXME: nested with other file, replace reason with subdiag.
    pub(crate) fn cannot_borrow_path_as_mutable_because(
        &self,
        span: Span,
        path: &str,
        reason: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        struct_span_err!(self, span, E0596, "cannot borrow {} as mutable{}", path, reason,)
    }

    pub(crate) fn cannot_mutate_in_immutable_section(
        &self,
        mutate_span: Span,
        immutable_span: Span,
        immutable_place: &str,
        immutable_section: &str,
        action: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(MutateInImmute {
            action,
            immutable_place,
            immutable_section,
            mutate_span,
            immutable_span,
        })
    }

    pub(crate) fn cannot_borrow_across_generator_yield(
        &self,
        span: Span,
        yield_span: Span,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(BorrowAcrossGeneratorYield { span, yield_span })
    }

    pub(crate) fn cannot_borrow_across_destructor(
        &self,
        borrow_span: Span,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(BorrowAcrossDestructor { borrow_span })
    }

    pub(crate) fn path_does_not_live_long_enough(
        &self,
        span: Span,
        path: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(PathShortLive { path, span })
    }

    pub(crate) fn cannot_return_reference_to_local(
        &self,
        span: Span,
        return_kind: &str,
        reference_desc: &str,
        path_desc: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(ReturnRefLocalErr {
            return_kind,
            reference: reference_desc,
            local: path_desc,
            span,
        })
    }

    pub(crate) fn cannot_capture_in_long_lived_closure(
        &self,
        closure_span: Span,
        closure_kind: &str,
        borrowed_path: &str,
        capture_span: Span,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(ClosureVarOutliveErr {
            closure_kind,
            borrowed_path,
            closure_span,
            capture_span,
        })
    }

    pub(crate) fn thread_local_value_does_not_live_long_enough(
        &self,
        span: Span,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(ThreadLocalOutliveErr { span })
    }

    pub(crate) fn temporary_value_borrowed_for_too_long(
        &self,
        span: Span,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(TemporaryDroppedErr { span })
    }

    #[rustc_lint_diagnostics]
    pub(crate) fn struct_span_err_with_code<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        self.infcx.tcx.sess.struct_span_err_with_code(sp, msg, code)
    }
}

pub(crate) fn borrowed_data_escapes_closure<'tcx>(
    tcx: TyCtxt<'tcx>,
    escape_span: Span,
    escapes_from: &str,
) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
    tcx.sess.create_err(BorrowEscapeClosure { escapes_from, escape_span })
}
