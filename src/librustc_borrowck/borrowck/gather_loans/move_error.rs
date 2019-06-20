use crate::borrowck::BorrowckCtxt;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::middle::mem_categorization::NoteClosureEnv;
use rustc::middle::mem_categorization::InteriorOffsetKind as Kind;
use rustc::ty;
use rustc_mir::util::borrowck_errors::{BorrowckErrors, Origin};
use syntax::ast;
use syntax_pos;
use errors::{DiagnosticBuilder, Applicability};
use crate::borrowck::gather_loans::gather_moves::PatternSource;
use log::debug;

pub struct MoveErrorCollector<'tcx> {
    errors: Vec<MoveError<'tcx>>
}

impl<'tcx> MoveErrorCollector<'tcx> {
    pub fn new() -> MoveErrorCollector<'tcx> {
        MoveErrorCollector {
            errors: Vec::new()
        }
    }

    pub fn add_error(&mut self, error: MoveError<'tcx>) {
        self.errors.push(error);
    }

    pub fn report_potential_errors<'a>(&self, bccx: &BorrowckCtxt<'a, 'tcx>) {
        report_move_errors(bccx, &self.errors)
    }
}

pub struct MoveError<'tcx> {
    move_from: mc::cmt<'tcx>,
    move_to: Option<MovePlace<'tcx>>
}

impl<'tcx> MoveError<'tcx> {
    pub fn with_move_info(move_from: mc::cmt<'tcx>,
                          move_to: Option<MovePlace<'tcx>>)
                          -> MoveError<'tcx> {
        MoveError {
            move_from,
            move_to,
        }
    }
}

#[derive(Clone)]
pub struct MovePlace<'tcx> {
    pub span: syntax_pos::Span,
    pub name: ast::Name,
    pub pat_source: PatternSource<'tcx>,
}

pub struct GroupedMoveErrors<'tcx> {
    move_from: mc::cmt<'tcx>,
    move_to_places: Vec<MovePlace<'tcx>>
}

fn report_move_errors<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>, errors: &[MoveError<'tcx>]) {
    let grouped_errors = group_errors_with_same_origin(errors);
    for error in &grouped_errors {
        let mut err = report_cannot_move_out_of(bccx, error.move_from.clone());
        let mut is_first_note = true;
        match error.move_to_places.get(0) {
            Some(&MovePlace { pat_source: PatternSource::LetDecl(ref e), .. }) => {
                // ignore patterns that are found at the top-level of a `let`;
                // see `get_pattern_source()` for details
                let initializer =
                    e.init.as_ref().expect("should have an initializer to get an error");
                if let Ok(snippet) = bccx.tcx.sess.source_map().span_to_snippet(initializer.span) {
                    err.span_suggestion(
                        initializer.span,
                        "consider using a reference instead",
                        format!("&{}", snippet),
                        Applicability::MaybeIncorrect // using a reference may not be the right fix
                    );
                }
            }
            _ => {
                for move_to in &error.move_to_places {

                    err = note_move_destination(err, move_to.span, move_to.name, is_first_note);
                    is_first_note = false;
                }
            }
        }
        if let NoteClosureEnv(upvar_id) = error.move_from.note {
            err.span_label(bccx.tcx.hir().span(upvar_id.var_path.hir_id),
                           "captured outer variable");
        }
        err.emit();
        bccx.signal_error();
    }
}

fn group_errors_with_same_origin<'tcx>(errors: &[MoveError<'tcx>])
                                       -> Vec<GroupedMoveErrors<'tcx>> {
    let mut grouped_errors = Vec::new();
    for error in errors {
        append_to_grouped_errors(&mut grouped_errors, error)
    }
    return grouped_errors;

    fn append_to_grouped_errors<'tcx>(grouped_errors: &mut Vec<GroupedMoveErrors<'tcx>>,
                                      error: &MoveError<'tcx>) {
        let move_from_id = error.move_from.hir_id;
        debug!("append_to_grouped_errors(move_from_id={:?})", move_from_id);
        let move_to = if error.move_to.is_some() {
            vec![error.move_to.clone().unwrap()]
        } else {
            Vec::new()
        };
        for ge in &mut *grouped_errors {
            if move_from_id == ge.move_from.hir_id && error.move_to.is_some() {
                debug!("appending move_to to list");
                ge.move_to_places.extend(move_to);
                return
            }
        }
        debug!("found a new move from location");
        grouped_errors.push(GroupedMoveErrors {
            move_from: error.move_from.clone(),
            move_to_places: move_to
        })
    }
}

// (keep in sync with gather_moves::check_and_get_illegal_move_origin )
fn report_cannot_move_out_of<'a, 'tcx>(bccx: &'a BorrowckCtxt<'a, 'tcx>,
                                       move_from: mc::cmt<'tcx>)
                                       -> DiagnosticBuilder<'a> {
    match move_from.cat {
        Categorization::Deref(_, mc::BorrowedPtr(..)) |
        Categorization::Deref(_, mc::UnsafePtr(..)) |
        Categorization::Deref(_, mc::Unique) |
        Categorization::ThreadLocal(..) |
        Categorization::StaticItem => {
            bccx.cannot_move_out_of(
                move_from.span, &move_from.descriptive_string(bccx.tcx), Origin::Ast)
        }
        Categorization::Interior(ref b, mc::InteriorElement(ik)) => {
            bccx.cannot_move_out_of_interior_noncopy(
                move_from.span, b.ty, Some(ik == Kind::Index), Origin::Ast)
        }

        Categorization::Downcast(ref b, _) |
        Categorization::Interior(ref b, mc::InteriorField(_)) => {
            match b.ty.sty {
                ty::Adt(def, _) if def.has_dtor(bccx.tcx) => {
                    bccx.cannot_move_out_of_interior_of_drop(
                        move_from.span, b.ty, Origin::Ast)
                }
                _ => {
                    span_bug!(move_from.span, "this path should not cause illegal move");
                }
            }
        }

        Categorization::Rvalue(..) |
        Categorization::Local(..) |
        Categorization::Upvar(..) => {
            span_bug!(move_from.span, "this path should not cause illegal move");
        }
    }
}

fn note_move_destination(mut err: DiagnosticBuilder<'_>,
                         move_to_span: syntax_pos::Span,
                         pat_name: ast::Name,
                         is_first_note: bool) -> DiagnosticBuilder<'_> {
    if is_first_note {
        err.span_label(
            move_to_span,
            format!("hint: to prevent move, use `ref {0}` or `ref mut {0}`",
                     pat_name));
        err
    } else {
        err.span_label(move_to_span,
                      format!("...and here (use `ref {0}` or `ref mut {0}`)",
                               pat_name));
        err
    }
}
