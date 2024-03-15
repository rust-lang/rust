#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use rustc_errors::{codes::*, struct_span_code_err, Diag, DiagCtxt};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;

use crate::{
    diagnostics::PlaceAndReason,
    session_diagnostics::{
        AssignBorrowErr, AssignErr, BorrowAcrossCoroutineYield, BorrowAcrossDestructor,
        InteriorDropMoveErr, MutBorrowErr, PathShortLive, UseMutBorrowErr,
    },
};

impl<'cx, 'tcx> crate::MirBorrowckCtxt<'cx, 'tcx> {
    pub fn dcx(&self) -> &'tcx DiagCtxt {
        self.infcx.dcx()
    }

    pub(crate) fn cannot_move_when_borrowed(
        &self,
        span: Span,
        borrow_span: Span,
        place: &str,
        borrow_place: &str,
        value_place: &str,
    ) -> Diag<'tcx> {
        self.dcx().create_err(crate::session_diagnostics::MoveBorrow {
            place,
            span,
            borrow_place,
            value_place,
            borrow_span,
        })
    }

    pub(crate) fn cannot_use_when_mutably_borrowed(
        &self,
        span: Span,
        desc: &str,
        borrow_span: Span,
        borrow_desc: &str,
    ) -> Diag<'tcx> {
        self.dcx().create_err(UseMutBorrowErr { desc, borrow_desc, span, borrow_span })
    }

    pub(crate) fn cannot_mutably_borrow_multiply(
        &self,
        new_loan_span: Span,
        new_place_name: &str,
        place: &str,
        old_loan_span: Span,
        old_place: &str,
        old_load_end_span: Option<Span>,
    ) -> Diag<'tcx> {
        use crate::session_diagnostics::MutBorrowMulti::*;
        let via = place.is_empty();
        self.dcx().create_err(if old_loan_span == new_loan_span {
            // Both borrows are happening in the same place
            // Meaning the borrow is occurring in a loop
            SameSpan {
                new_place_name,
                place,
                old_place,
                is_place_empty: via,
                new_loan_span,
                old_load_end_span,
                eager_label: crate::session_diagnostics::MutMultiLoopLabel {
                    new_place_name,
                    place,
                    is_place_empty: via,
                    new_loan_span,
                },
            }
        } else {
            ChangedSpan {
                new_place_name,
                place,
                old_place,
                is_place_empty: via,
                is_old_place_empty: old_place.is_empty(),
                new_loan_span,
                old_loan_span,
                old_load_end_span,
            }
        })
    }

    pub(crate) fn cannot_uniquely_borrow_by_two_closures(
        &self,
        new_loan_span: Span,
        desc: &str,
        old_loan_span: Span,
        old_load_end_span: Option<Span>,
    ) -> Diag<'tcx> {
        use crate::session_diagnostics::ClosureConstructLabel::*;
        let (case, diff_span) = if old_loan_span == new_loan_span {
            (Both { old_loan_span }, None)
        } else {
            (First { old_loan_span }, Some(new_loan_span))
        };
        self.dcx().create_err(crate::session_diagnostics::TwoClosuresUniquelyBorrowErr {
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
    ) -> Diag<'tcx> {
        self.dcx().create_err(crate::session_diagnostics::ClosureUniquelyBorrowErr {
            new_loan_span,
            container_name,
            desc_new,
            opt_via,
            old_loan_span,
            noun_old,
            old_opt_via,
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
    ) -> Diag<'tcx> {
        self.dcx().create_err(crate::session_diagnostics::ClosureReBorrowErr {
            new_loan_span,
            container_name,
            desc_new,
            opt_via,
            kind_new,
            old_loan_span,
            old_opt_via,
            previous_end_span,
            second_borrow_desc,
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
    ) -> Diag<'tcx> {
        use crate::session_diagnostics::BorrowOccurLabel::*;
        let via = |msg: &str| msg.is_empty();
        let (new_occur, old_occur) = if msg_new == "" {
            // If `msg_new` is empty, then this isn't a borrow of a union field.
            (Here { span, kind: kind_new }, Here { span: old_span, kind: kind_old })
        } else {
            // If `msg_new` isn't empty, then this a borrow of a union field.
            (
                HereOverlap { span, kind_new, msg_new, msg_old },
                HereVia { span: old_span, kind_old, is_msg_old_empty: via(msg_old), msg_old },
            )
        };
        self.dcx().create_err(crate::session_diagnostics::ReborrowBorrowedErr {
            desc_new,
            is_msg_new_empty: via(msg_new),
            msg_new,
            kind_new,
            noun_old,
            kind_old,
            is_msg_old_empty: via(msg_old),
            msg_old,
            span,
            old_load_end_span,
            new_occur,
            old_occur,
        })
    }

    pub(crate) fn cannot_assign_to_borrowed(
        &self,
        span: Span,
        borrow_span: Span,
        desc: &str,
    ) -> Diag<'tcx> {
        self.dcx().create_err(AssignBorrowErr { desc, span, borrow_span })
    }

    pub(crate) fn cannot_reassign_immutable(
        &self,
        span: Span,
        desc: &str,
        is_arg: bool,
    ) -> Diag<'tcx> {
        use crate::session_diagnostics::ReassignImmut::*;
        self.dcx().create_err(if is_arg {
            Arg { span, place: desc }
        } else {
            Var { span, place: desc }
        })
    }

    pub(crate) fn cannot_assign(
        &self,
        span: Span,
        path_and_reason: PlaceAndReason<'_>,
    ) -> Diag<'tcx> {
        let diag = match path_and_reason {
            PlaceAndReason::DeclaredImmute(place, name) => {
                if let Some(name) = name {
                    AssignErr::SymbolDeclaredImmute { span, place, name }
                } else {
                    AssignErr::PlaceDeclaredImmute { span, place }
                }
            }
            PlaceAndReason::InPatternGuard(place) => AssignErr::PatternGuardImmute { span, place },
            PlaceAndReason::StaticItem(place, name) => {
                if let Some(name) = name {
                    AssignErr::SymbolStatic { span, place, static_name: name }
                } else {
                    AssignErr::PlaceStatic { span, place }
                }
            }
            PlaceAndReason::UpvarCaptured(place) => AssignErr::UpvarInFn { span, place },
            PlaceAndReason::SelfCaptured(place) => AssignErr::CapturedInFn { span, place },
            PlaceAndReason::BehindPointer(place, pointer_ty, name) => {
                if place.0.is_some() {
                    match pointer_ty {
                        crate::diagnostics::BorrowedContentSource::DerefRawPointer => {
                            AssignErr::PlaceBehindRawPointer { span, place }
                        }
                        crate::diagnostics::BorrowedContentSource::DerefMutableRef => {
                            unreachable!()
                        }
                        crate::diagnostics::BorrowedContentSource::DerefSharedRef => {
                            AssignErr::PlaceBehindSharedRef { span, place }
                        }
                        crate::diagnostics::BorrowedContentSource::OverloadedDeref(_) => {
                            AssignErr::PlaceBehindDeref { span, place, name }
                        }
                        crate::diagnostics::BorrowedContentSource::OverloadedIndex(_) => {
                            AssignErr::PlaceBehindIndex { span, place, name }
                        }
                    }
                } else {
                    match pointer_ty {
                        crate::diagnostics::BorrowedContentSource::DerefRawPointer => {
                            AssignErr::DataBehindRawPointer { span }
                        }
                        crate::diagnostics::BorrowedContentSource::DerefMutableRef => {
                            unreachable!()
                        }
                        crate::diagnostics::BorrowedContentSource::DerefSharedRef => {
                            AssignErr::DataBehindSharedRef { span }
                        }
                        crate::diagnostics::BorrowedContentSource::OverloadedDeref(_) => {
                            AssignErr::DataBehindDeref { span, name }
                        }
                        crate::diagnostics::BorrowedContentSource::OverloadedIndex(_) => {
                            AssignErr::DataBehindIndex { span, name }
                        }
                    }
                }
            }
        };
        self.dcx().create_err(diag)
    }

    pub(crate) fn cannot_move_out_of(
        &self,
        move_from_span: Span,
        move_from_desc: &str,
    ) -> Diag<'tcx> {
        struct_span_code_err!(
            self.dcx(),
            move_from_span,
            E0507,
            "cannot move out of {}",
            move_from_desc
        )
    }

    /// Signal an error due to an attempt to move out of the interior
    /// of an array or slice. `is_index` is None when error origin
    /// didn't capture whether there was an indexing operation or not.
    pub(crate) fn cannot_move_out_of_interior_noncopy(
        &self,
        move_from_span: Span,
        ty: Ty<'_>,
        is_index: Option<bool>,
    ) -> Diag<'tcx> {
        let type_name = match (&ty.kind(), is_index) {
            (&ty::Array(_, _), Some(true)) | (&ty::Array(_, _), None) => "array",
            (&ty::Slice(_), _) => "slice",
            _ => span_bug!(move_from_span, "this path should not cause illegal move"),
        };
        struct_span_code_err!(
            self.dcx(),
            move_from_span,
            E0508,
            "cannot move out of type `{}`, a non-copy {}",
            ty,
            type_name,
        )
        .with_span_label(move_from_span, "cannot move out of here")
    }

    pub(crate) fn cannot_move_out_of_interior_of_drop(
        &self,
        move_from_span: Span,
        container_ty: Ty<'_>,
    ) -> Diag<'tcx> {
        self.dcx().create_err(InteriorDropMoveErr { container_ty, move_from_span })
    }

    pub(crate) fn cannot_act_on_moved_value(
        &self,
        use_span: Span,
        verb: &str,
        optional_adverb_for_moved: &str,
        moved_path: Option<String>,
    ) -> Diag<'tcx> {
        let moved_path = moved_path.map(|mp| format!(": `{mp}`")).unwrap_or_default();

        struct_span_code_err!(
            self.dcx(),
            use_span,
            E0382,
            "{} of {}moved value{}",
            verb,
            optional_adverb_for_moved,
            moved_path,
        )
    }

    pub(crate) fn cannot_borrow_path_as_mutable_because(
        &self,
        span: Span,
        path_and_reason: PlaceAndReason<'_>,
    ) -> Diag<'tcx> {
        let diag = match path_and_reason {
            PlaceAndReason::DeclaredImmute(place, name) => {
                if let Some(name) = name {
                    MutBorrowErr::SymbolDeclaredImmute { span, place, name }
                } else {
                    MutBorrowErr::PlaceDeclaredImmute { span, place }
                }
            }
            PlaceAndReason::InPatternGuard(place) => {
                MutBorrowErr::PatternGuardImmute { span, place }
            }
            PlaceAndReason::StaticItem(place, name) => {
                if let Some(name) = name {
                    MutBorrowErr::SymbolStatic { span, place, static_name: name }
                } else {
                    MutBorrowErr::PlaceStatic { span, place }
                }
            }
            PlaceAndReason::UpvarCaptured(place) => MutBorrowErr::UpvarInFn { span, place },
            PlaceAndReason::SelfCaptured(place) => MutBorrowErr::CapturedInFn { span, place },
            PlaceAndReason::BehindPointer(place, pointer_ty, name) => {
                if place.0.is_some() {
                    match pointer_ty {
                        crate::diagnostics::BorrowedContentSource::DerefRawPointer => {
                            MutBorrowErr::SelfBehindRawPointer { span, place }
                        }
                        crate::diagnostics::BorrowedContentSource::DerefSharedRef => {
                            MutBorrowErr::SelfBehindSharedRef { span, place }
                        }
                        crate::diagnostics::BorrowedContentSource::OverloadedDeref(_) => {
                            MutBorrowErr::SelfBehindDeref { span, place, name }
                        }
                        crate::diagnostics::BorrowedContentSource::OverloadedIndex(_) => {
                            MutBorrowErr::SelfBehindIndex { span, place, name }
                        }
                        crate::diagnostics::BorrowedContentSource::DerefMutableRef => {
                            unreachable!()
                        }
                    }
                } else {
                    match pointer_ty {
                        crate::diagnostics::BorrowedContentSource::DerefRawPointer => {
                            MutBorrowErr::DataBehindRawPointer { span }
                        }
                        crate::diagnostics::BorrowedContentSource::DerefMutableRef => {
                            unreachable!()
                        }
                        crate::diagnostics::BorrowedContentSource::DerefSharedRef => {
                            MutBorrowErr::DataBehindSharedRef { span }
                        }
                        crate::diagnostics::BorrowedContentSource::OverloadedDeref(_) => {
                            MutBorrowErr::DataBehindDeref { span, name }
                        }
                        crate::diagnostics::BorrowedContentSource::OverloadedIndex(_) => {
                            MutBorrowErr::DataBehindIndex { span, name }
                        }
                    }
                }
            }
        };
        self.dcx().create_err(diag)
    }

    pub(crate) fn cannot_mutate_in_immutable_section(
        &self,
        mutate_span: Span,
        immutable_span: Span,
        immutable_place: &str,
        immutable_section: &str,
        action: &str,
    ) -> Diag<'tcx> {
        struct_span_code_err!(
            self.dcx(),
            mutate_span,
            E0510,
            "cannot {} {} in {}",
            action,
            immutable_place,
            immutable_section,
        )
        .with_span_label(mutate_span, format!("cannot {action}"))
        .with_span_label(immutable_span, format!("value is immutable in {immutable_section}"))
    }

    pub(crate) fn cannot_borrow_across_coroutine_yield(
        &self,
        span: Span,
        yield_span: Span,
    ) -> Diag<'tcx> {
        let coroutine_kind = self.body.coroutine.as_ref().unwrap().coroutine_kind;
        self.dcx().create_err(BorrowAcrossCoroutineYield {
            span,
            yield_span,
            coroutine_kind: format!("{coroutine_kind:#}"),
        })
    }

    pub(crate) fn cannot_borrow_across_destructor(&self, borrow_span: Span) -> Diag<'tcx> {
        self.dcx().create_err(BorrowAcrossDestructor { borrow_span })
    }

    pub(crate) fn path_does_not_live_long_enough(&self, span: Span, path: &str) -> Diag<'tcx> {
        self.dcx().create_err(PathShortLive { path, span })
    }

    pub(crate) fn cannot_return_reference_to_local(
        &self,
        span: Span,
        return_kind: &str,
        reference_desc: &str,
        path_desc: &str,
    ) -> Diag<'tcx> {
        struct_span_code_err!(
            self.dcx(),
            span,
            E0515,
            "cannot {RETURN} {REFERENCE} {LOCAL}",
            RETURN = return_kind,
            REFERENCE = reference_desc,
            LOCAL = path_desc,
        )
        .with_span_label(
            span,
            format!("{return_kind}s a {reference_desc} data owned by the current function"),
        )
    }

    pub(crate) fn cannot_capture_in_long_lived_closure(
        &self,
        closure_span: Span,
        closure_kind: &str,
        borrowed_path: &str,
        capture_span: Span,
        scope: &str,
    ) -> Diag<'tcx> {
        struct_span_code_err!(
            self.dcx(),
            closure_span,
            E0373,
            "{closure_kind} may outlive the current {scope}, but it borrows {borrowed_path}, \
             which is owned by the current {scope}",
        )
        .with_span_label(capture_span, format!("{borrowed_path} is borrowed here"))
        .with_span_label(closure_span, format!("may outlive borrowed value {borrowed_path}"))
    }

    pub(crate) fn thread_local_value_does_not_live_long_enough(&self, span: Span) -> Diag<'tcx> {
        struct_span_code_err!(
            self.dcx(),
            span,
            E0712,
            "thread-local variable borrowed past end of function",
        )
    }

    pub(crate) fn temporary_value_borrowed_for_too_long(&self, span: Span) -> Diag<'tcx> {
        struct_span_code_err!(self.dcx(), span, E0716, "temporary value dropped while borrowed",)
    }
}

pub(crate) fn borrowed_data_escapes_closure<'tcx>(
    tcx: TyCtxt<'tcx>,
    escape_span: Span,
    escapes_from: &str,
) -> Diag<'tcx> {
    struct_span_code_err!(
        tcx.dcx(),
        escape_span,
        E0521,
        "borrowed data escapes outside of {}",
        escapes_from,
    )
}
