use rustc::session::config::BorrowckMode;
use rustc::ty::{self, Ty, TyCtxt};
use rustc_errors::{DiagnosticBuilder, DiagnosticId};
use syntax_pos::{MultiSpan, Span};

use std::fmt;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Origin {
    Ast,
    Mir,
}

impl fmt::Display for Origin {
    fn fmt(&self, _w: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME(chrisvittal) remove Origin entirely
        // Print no origin info
        Ok(())
    }
}

impl Origin {
    /// Whether we should emit errors for the origin in the given mode
    pub fn should_emit_errors(self, mode: BorrowckMode) -> bool {
        match self {
            Origin::Ast => mode.use_ast(),
            Origin::Mir => true,
        }
    }
}

pub trait BorrowckErrors<'cx>: Sized + Copy {
    fn struct_span_err_with_code<S: Into<MultiSpan>>(
        self,
        sp: S,
        msg: &str,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'cx>;

    fn struct_span_err<S: Into<MultiSpan>>(self, sp: S, msg: &str) -> DiagnosticBuilder<'cx>;

    /// Cancels the given error if we shouldn't emit errors for a given
    /// origin in the current mode.
    ///
    /// Always make sure that the error gets passed through this function
    /// before you return it.
    fn cancel_if_wrong_origin(
        self,
        diag: DiagnosticBuilder<'cx>,
        o: Origin,
    ) -> DiagnosticBuilder<'cx>;

    fn cannot_move_when_borrowed(
        self,
        span: Span,
        desc: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(
            self,
            span,
            E0505,
            "cannot move out of `{}` because it is borrowed{OGN}",
            desc,
            OGN = o
        );
        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_use_when_mutably_borrowed(
        self,
        span: Span,
        desc: &str,
        borrow_span: Span,
        borrow_desc: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let mut err = struct_span_err!(
            self,
            span,
            E0503,
            "cannot use `{}` because it was mutably borrowed{OGN}",
            desc,
            OGN = o
        );

        err.span_label(
            borrow_span,
            format!("borrow of `{}` occurs here", borrow_desc),
        );
        err.span_label(span, format!("use of borrowed `{}`", borrow_desc));

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_act_on_uninitialized_variable(
        self,
        span: Span,
        verb: &str,
        desc: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(
            self,
            span,
            E0381,
            "{} of possibly uninitialized variable: `{}`{OGN}",
            verb,
            desc,
            OGN = o
        );
        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_mutably_borrow_multiply(
        self,
        new_loan_span: Span,
        desc: &str,
        opt_via: &str,
        old_loan_span: Span,
        old_opt_via: &str,
        old_load_end_span: Option<Span>,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let via = |msg: &str|
            if msg.is_empty() { msg.to_string() } else { format!(" (via `{}`)", msg) };
        let mut err = struct_span_err!(
            self,
            new_loan_span,
            E0499,
            "cannot borrow `{}`{} as mutable more than once at a time{OGN}",
            desc,
            via(opt_via),
            OGN = o
        );
        if old_loan_span == new_loan_span {
            // Both borrows are happening in the same place
            // Meaning the borrow is occurring in a loop
            err.span_label(
                new_loan_span,
                format!(
                    "mutable borrow starts here in previous \
                     iteration of loop{}",
                    opt_via
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
        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_uniquely_borrow_by_two_closures(
        self,
        new_loan_span: Span,
        desc: &str,
        old_loan_span: Span,
        old_load_end_span: Option<Span>,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let mut err = struct_span_err!(
            self,
            new_loan_span,
            E0524,
            "two closures require unique access to `{}` at the same time{OGN}",
            desc,
            OGN = o
        );
        if old_loan_span == new_loan_span {
            err.span_label(
                old_loan_span,
                "closures are constructed here in different iterations of loop"
            );
        } else {
            err.span_label(old_loan_span, "first closure is constructed here");
            err.span_label(new_loan_span, "second closure is constructed here");
        }
        if let Some(old_load_end_span) = old_load_end_span {
            err.span_label(old_load_end_span, "borrow from first closure ends here");
        }
        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_uniquely_borrow_by_one_closure(
        self,
        new_loan_span: Span,
        container_name: &str,
        desc_new: &str,
        opt_via: &str,
        old_loan_span: Span,
        noun_old: &str,
        old_opt_via: &str,
        previous_end_span: Option<Span>,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let mut err = struct_span_err!(
            self,
            new_loan_span,
            E0500,
            "closure requires unique access to `{}` but {} is already borrowed{}{OGN}",
            desc_new,
            noun_old,
            old_opt_via,
            OGN = o
        );
        err.span_label(
            new_loan_span,
            format!("{} construction occurs here{}", container_name, opt_via),
        );
        err.span_label(old_loan_span, format!("borrow occurs here{}", old_opt_via));
        if let Some(previous_end_span) = previous_end_span {
            err.span_label(previous_end_span, "borrow ends here");
        }
        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_reborrow_already_uniquely_borrowed(
        self,
        new_loan_span: Span,
        container_name: &str,
        desc_new: &str,
        opt_via: &str,
        kind_new: &str,
        old_loan_span: Span,
        old_opt_via: &str,
        previous_end_span: Option<Span>,
        second_borrow_desc: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let mut err = struct_span_err!(
            self,
            new_loan_span,
            E0501,
            "cannot borrow `{}`{} as {} because previous closure \
             requires unique access{OGN}",
            desc_new,
            opt_via,
            kind_new,
            OGN = o
        );
        err.span_label(
            new_loan_span,
            format!("{}borrow occurs here{}", second_borrow_desc, opt_via),
        );
        err.span_label(
            old_loan_span,
            format!("{} construction occurs here{}", container_name, old_opt_via),
        );
        if let Some(previous_end_span) = previous_end_span {
            err.span_label(previous_end_span, "borrow from closure ends here");
        }
        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_reborrow_already_borrowed(
        self,
        span: Span,
        desc_new: &str,
        msg_new: &str,
        kind_new: &str,
        old_span: Span,
        noun_old: &str,
        kind_old: &str,
        msg_old: &str,
        old_load_end_span: Option<Span>,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let via = |msg: &str|
            if msg.is_empty() { msg.to_string() } else { format!(" (via `{}`)", msg) };
        let mut err = struct_span_err!(
            self,
            span,
            E0502,
            "cannot borrow `{}`{} as {} because {} is also borrowed \
             as {}{}{OGN}",
            desc_new,
            via(msg_new),
            kind_new,
            noun_old,
            kind_old,
            via(msg_old),
            OGN = o
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
                    "{} borrow of `{}` -- which overlaps with `{}` -- occurs here",
                    kind_new, msg_new, msg_old,
                )
            );
            err.span_label(
                old_span,
                format!("{} borrow occurs here{}", kind_old, via(msg_old)),
            );
        }

        if let Some(old_load_end_span) = old_load_end_span {
            err.span_label(old_load_end_span, format!("{} borrow ends here", kind_old));
        }

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_assign_to_borrowed(
        self,
        span: Span,
        borrow_span: Span,
        desc: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let mut err = struct_span_err!(
            self,
            span,
            E0506,
            "cannot assign to `{}` because it is borrowed{OGN}",
            desc,
            OGN = o
        );

        err.span_label(borrow_span, format!("borrow of `{}` occurs here", desc));
        err.span_label(
            span,
            format!("assignment to borrowed `{}` occurs here", desc),
        );

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_move_into_closure(self, span: Span, desc: &str, o: Origin) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(
            self,
            span,
            E0504,
            "cannot move `{}` into closure because it is borrowed{OGN}",
            desc,
            OGN = o
        );

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_reassign_immutable(
        self,
        span: Span,
        desc: &str,
        is_arg: bool,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let msg = if is_arg {
            "to immutable argument"
        } else {
            "twice to immutable variable"
        };
        let err = struct_span_err!(
            self,
            span,
            E0384,
            "cannot assign {} `{}`{OGN}",
            msg,
            desc,
            OGN = o
        );

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_assign(self, span: Span, desc: &str, o: Origin) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(self, span, E0594, "cannot assign to {}{OGN}", desc, OGN = o);
        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_assign_static(self, span: Span, desc: &str, o: Origin) -> DiagnosticBuilder<'cx> {
        self.cannot_assign(span, &format!("immutable static item `{}`", desc), o)
    }

    fn cannot_move_out_of(
        self,
        move_from_span: Span,
        move_from_desc: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(
            self,
            move_from_span,
            E0507,
            "cannot move out of {}{OGN}",
            move_from_desc,
            OGN = o
        );

        self.cancel_if_wrong_origin(err, o)
    }

    /// Signal an error due to an attempt to move out of the interior
    /// of an array or slice. `is_index` is None when error origin
    /// didn't capture whether there was an indexing operation or not.
    fn cannot_move_out_of_interior_noncopy(
        self,
        move_from_span: Span,
        ty: Ty<'_>,
        is_index: Option<bool>,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let type_name = match (&ty.sty, is_index) {
            (&ty::Array(_, _), Some(true)) | (&ty::Array(_, _), None) => "array",
            (&ty::Slice(_), _) => "slice",
            _ => span_bug!(move_from_span, "this path should not cause illegal move"),
        };
        let mut err = struct_span_err!(
            self,
            move_from_span,
            E0508,
            "cannot move out of type `{}`, a non-copy {}{OGN}",
            ty,
            type_name,
            OGN = o
        );
        err.span_label(move_from_span, "cannot move out of here");

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_move_out_of_interior_of_drop(
        self,
        move_from_span: Span,
        container_ty: Ty<'_>,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let mut err = struct_span_err!(
            self,
            move_from_span,
            E0509,
            "cannot move out of type `{}`, which implements the `Drop` trait{OGN}",
            container_ty,
            OGN = o
        );
        err.span_label(move_from_span, "cannot move out of here");

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_act_on_moved_value(
        self,
        use_span: Span,
        verb: &str,
        optional_adverb_for_moved: &str,
        moved_path: Option<String>,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let moved_path = moved_path
            .map(|mp| format!(": `{}`", mp))
            .unwrap_or_default();

        let err = struct_span_err!(
            self,
            use_span,
            E0382,
            "{} of {}moved value{}{OGN}",
            verb,
            optional_adverb_for_moved,
            moved_path,
            OGN = o
        );

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_partially_reinit_an_uninit_struct(
        self,
        span: Span,
        uninit_path: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(
            self,
            span,
            E0383,
            "partial reinitialization of uninitialized structure `{}`{OGN}",
            uninit_path,
            OGN = o
        );

        self.cancel_if_wrong_origin(err, o)
    }

    fn closure_cannot_assign_to_borrowed(
        self,
        span: Span,
        descr: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(
            self,
            span,
            E0595,
            "closure cannot assign to {}{OGN}",
            descr,
            OGN = o
        );

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_borrow_path_as_mutable_because(
        self,
        span: Span,
        path: &str,
        reason: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(
            self,
            span,
            E0596,
            "cannot borrow {} as mutable{}{OGN}",
            path,
            reason,
            OGN = o,
        );

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_borrow_path_as_mutable(
        self,
        span: Span,
        path: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        self.cannot_borrow_path_as_mutable_because(span, path, "", o)
    }

    fn cannot_mutate_in_match_guard(
        self,
        mutate_span: Span,
        match_span: Span,
        match_place: &str,
        action: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let mut err = struct_span_err!(
            self,
            mutate_span,
            E0510,
            "cannot {} `{}` in match guard{OGN}",
            action,
            match_place,
            OGN = o
        );
        err.span_label(mutate_span, format!("cannot {}", action));
        err.span_label(match_span, String::from("value is immutable in match guard"));

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_borrow_across_generator_yield(
        self,
        span: Span,
        yield_span: Span,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let mut err = struct_span_err!(
            self,
            span,
            E0626,
            "borrow may still be in use when generator yields{OGN}",
            OGN = o
        );
        err.span_label(yield_span, "possible yield occurs here");

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_borrow_across_destructor(
        self,
        borrow_span: Span,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(
            self,
            borrow_span,
            E0713,
            "borrow may still be in use when destructor runs{OGN}",
            OGN = o
        );

        self.cancel_if_wrong_origin(err, o)
    }

    fn path_does_not_live_long_enough(
        self,
        span: Span,
        path: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(
            self,
            span,
            E0597,
            "{} does not live long enough{OGN}",
            path,
            OGN = o
        );

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_return_reference_to_local(
        self,
        span: Span,
        return_kind: &str,
        reference_desc: &str,
        path_desc: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let mut err = struct_span_err!(
            self,
            span,
            E0515,
            "cannot {RETURN} {REFERENCE} {LOCAL}{OGN}",
            RETURN=return_kind,
            REFERENCE=reference_desc,
            LOCAL=path_desc,
            OGN = o
        );

        err.span_label(
            span,
            format!("{}s a {} data owned by the current function", return_kind, reference_desc),
        );

        self.cancel_if_wrong_origin(err, o)
    }

    fn lifetime_too_short_for_reborrow(
        self,
        span: Span,
        path: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(
            self,
            span,
            E0598,
            "lifetime of {} is too short to guarantee \
             its contents can be safely reborrowed{OGN}",
            path,
            OGN = o
        );

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_act_on_capture_in_sharable_fn(
        self,
        span: Span,
        bad_thing: &str,
        help: (Span, &str),
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let (help_span, help_msg) = help;
        let mut err = struct_span_err!(
            self,
            span,
            E0387,
            "{} in a captured outer variable in an `Fn` closure{OGN}",
            bad_thing,
            OGN = o
        );
        err.span_help(help_span, help_msg);

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_assign_into_immutable_reference(
        self,
        span: Span,
        bad_thing: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let mut err = struct_span_err!(
            self,
            span,
            E0389,
            "{} in a `&` reference{OGN}",
            bad_thing,
            OGN = o
        );
        err.span_label(span, "assignment into an immutable reference");

        self.cancel_if_wrong_origin(err, o)
    }

    fn cannot_capture_in_long_lived_closure(
        self,
        closure_span: Span,
        borrowed_path: &str,
        capture_span: Span,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let mut err = struct_span_err!(
            self,
            closure_span,
            E0373,
            "closure may outlive the current function, \
             but it borrows {}, \
             which is owned by the current function{OGN}",
            borrowed_path,
            OGN = o
        );
        err.span_label(capture_span, format!("{} is borrowed here", borrowed_path))
            .span_label(
                closure_span,
                format!("may outlive borrowed value {}", borrowed_path),
            );

        self.cancel_if_wrong_origin(err, o)
    }

    fn borrowed_data_escapes_closure(
        self,
        escape_span: Span,
        escapes_from: &str,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(
            self,
            escape_span,
            E0521,
            "borrowed data escapes outside of {}{OGN}",
            escapes_from,
            OGN = o
        );

        self.cancel_if_wrong_origin(err, o)
    }

    fn thread_local_value_does_not_live_long_enough(
        self,
        span: Span,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(
            self,
            span,
            E0712,
            "thread-local variable borrowed past end of function{OGN}",
            OGN = o
        );

        self.cancel_if_wrong_origin(err, o)
    }

    fn temporary_value_borrowed_for_too_long(
        self,
        span: Span,
        o: Origin,
    ) -> DiagnosticBuilder<'cx> {
        let err = struct_span_err!(
            self,
            span,
            E0716,
            "temporary value dropped while borrowed{OGN}",
            OGN = o
        );

        self.cancel_if_wrong_origin(err, o)
    }
}

impl BorrowckErrors<'tcx> for TyCtxt<'tcx> {
    fn struct_span_err_with_code<S: Into<MultiSpan>>(
        self,
        sp: S,
        msg: &str,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'tcx> {
        self.sess.struct_span_err_with_code(sp, msg, code)
    }

    fn struct_span_err<S: Into<MultiSpan>>(self, sp: S, msg: &str) -> DiagnosticBuilder<'tcx> {
        self.sess.struct_span_err(sp, msg)
    }

    fn cancel_if_wrong_origin(
        self,
        mut diag: DiagnosticBuilder<'tcx>,
        o: Origin,
    ) -> DiagnosticBuilder<'tcx> {
        if !o.should_emit_errors(self.borrowck_mode()) {
            self.sess.diagnostic().cancel(&mut diag);
        }
        diag
    }
}
