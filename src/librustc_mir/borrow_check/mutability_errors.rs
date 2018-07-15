// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir;
use rustc::mir::{self, BindingForm, ClearCrossCrate, Local, Location, Mir};
use rustc::mir::{Mutability, Place, Projection, ProjectionElem, Static};
use rustc::ty::{self, TyCtxt};
use rustc_data_structures::indexed_vec::Idx;
use syntax_pos::Span;

use borrow_check::MirBorrowckCtxt;
use util::borrowck_errors::{BorrowckErrors, Origin};
use util::collect_writes::FindAssignments;
use util::suggest_ref_mut;

#[derive(Copy, Clone, Debug)]
pub(super) enum AccessKind {
    MutableBorrow,
    Mutate,
}

impl<'a, 'gcx, 'tcx> MirBorrowckCtxt<'a, 'gcx, 'tcx> {
    pub(super) fn report_mutability_error(
        &mut self,
        access_place: &Place<'tcx>,
        span: Span,
        the_place_err: &Place<'tcx>,
        error_access: AccessKind,
        location: Location,
    ) {
        let mut err;
        let item_msg = match self.describe_place(place) {
            Some(name) => format!("immutable item `{}`", name),
            None => "immutable item".to_owned(),
        };

        // `act` and `acted_on` are strings that let us abstract over
        // the verbs used in some diagnostic messages.
        let act;
        let acted_on;

        match error_access {
            AccessKind::Mutate => {
                let item_msg = match the_place_err {
                    Place::Projection(box Projection {
                        base: _,
                        elem: ProjectionElem::Deref,
                    }) => match self.describe_place(place) {
                        Some(description) => {
                            format!("`{}` which is behind a `&` reference", description)
                        }
                        None => format!("data in a `&` reference"),
                    },
                    _ => item_msg,
                };
                err = self.tcx.cannot_assign(span, &item_msg, Origin::Mir);
                act = "assign";
                acted_on = "written";
            }
            AccessKind::MutableBorrow => {
                err = self
                    .tcx
                    .cannot_borrow_path_as_mutable(span, &item_msg, Origin::Mir);
                act = "borrow as mutable";
                acted_on = "borrowed as mutable";
            }
        }

        match the_place_err {
            // We want to suggest users use `let mut` for local (user
            // variable) mutations...
            Place::Local(local) if self.mir.local_decls[*local].can_be_made_mutable() => {
                // ... but it doesn't make sense to suggest it on
                // variables that are `ref x`, `ref mut x`, `&self`,
                // or `&mut self` (such variables are simply not
                // mutable)..
                let local_decl = &self.mir.local_decls[*local];
                assert_eq!(local_decl.mutability, Mutability::Not);

                err.span_label(span, format!("cannot {ACT}", ACT = act));
                err.span_suggestion(
                    local_decl.source_info.span,
                    "consider changing this to be mutable",
                    format!("mut {}", local_decl.name.unwrap()),
                );
            }

            // complete hack to approximate old AST-borrowck
            // diagnostic: if the span starts with a mutable borrow of
            // a local variable, then just suggest the user remove it.
            Place::Local(_)
                if {
                    if let Ok(snippet) = self.tcx.sess.codemap().span_to_snippet(span) {
                        snippet.starts_with("&mut ")
                    } else {
                        false
                    }
                } =>
            {
                err.span_label(span, format!("cannot {ACT}", ACT = act));
                err.span_label(span, "try removing `&mut` here");
            }

            // We want to point out when a `&` can be readily replaced
            // with an `&mut`.
            //
            // FIXME: can this case be generalized to work for an
            // arbitrary base for the projection?
            Place::Projection(box Projection {
                base: Place::Local(local),
                elem: ProjectionElem::Deref,
            }) if self.mir.local_decls[*local].is_user_variable.is_some() => {
                let local_decl = &self.mir.local_decls[*local];
                let suggestion = match local_decl.is_user_variable.as_ref().unwrap() {
                    ClearCrossCrate::Set(mir::BindingForm::ImplicitSelf) => {
                        Some(suggest_ampmut_self(local_decl))
                    },

                    ClearCrossCrate::Set(mir::BindingForm::Var(mir::VarBindingForm {
                        binding_mode: ty::BindingMode::BindByValue(_),
                        opt_ty_info,
                        ..
                    })) => Some(suggest_ampmut(
                        self.tcx,
                        self.mir,
                        *local,
                        local_decl,
                        *opt_ty_info,
                    )),

                    ClearCrossCrate::Set(mir::BindingForm::Var(mir::VarBindingForm {
                        binding_mode: ty::BindingMode::BindByReference(_),
                        ..
                    })) => suggest_ref_mut(self.tcx, local_decl.source_info.span),

                    ClearCrossCrate::Clear => bug!("saw cleared local state"),
                };

                if let Some((err_help_span, suggested_code)) = suggestion {
                    err.span_suggestion(
                        err_help_span,
                        "consider changing this to be a mutable reference",
                        suggested_code,
                    );
                }

                if let Some(name) = local_decl.name {
                    err.span_label(
                        span,
                        format!(
                            "`{NAME}` is a `&` reference, \
                             so the data it refers to cannot be {ACTED_ON}",
                            NAME = name,
                            ACTED_ON = acted_on
                        ),
                    );
                } else {
                    err.span_label(
                        span,
                        format!("cannot {ACT} through `&`-reference", ACT = act),
                    );
                }
            }

            _ => {
                err.span_label(span, format!("cannot {ACT}", ACT = act));
            }
        }

        err.emit();
    }

    // Does this place refer to what the user sees as an upvar
    fn is_upvar(&self, place: &Place<'tcx>) -> bool {
        match *place {
            Place::Projection(box Projection {
                ref base,
                elem: ProjectionElem::Field(_, _),
            }) => {
                let base_ty = base.ty(self.mir, self.tcx).to_ty(self.tcx);
                is_closure_or_generator(base_ty)
            }
            Place::Projection(box Projection {
                base:
                    Place::Projection(box Projection {
                        ref base,
                        elem: ProjectionElem::Field(upvar_index, _),
                    }),
                elem: ProjectionElem::Deref,
            }) => {
                let base_ty = base.ty(self.mir, self.tcx).to_ty(self.tcx);
                is_closure_or_generator(base_ty) && self.mir.upvar_decls[upvar_index.index()].by_ref
            }
            _ => false,
        }
    }
}

fn suggest_ampmut_self<'cx, 'gcx, 'tcx>(local_decl: &mir::LocalDecl<'tcx>) -> (Span, String) {
    (local_decl.source_info.span, "&mut self".to_string())
}

// When we want to suggest a user change a local variable to be a `&mut`, there
// are three potential "obvious" things to highlight:
//
// let ident [: Type] [= RightHandSideExpression];
//     ^^^^^    ^^^^     ^^^^^^^^^^^^^^^^^^^^^^^
//     (1.)     (2.)              (3.)
//
// We can always fallback on highlighting the first. But chances are good that
// the user experience will be better if we highlight one of the others if possible;
// for example, if the RHS is present and the Type is not, then the type is going to
// be inferred *from* the RHS, which means we should highlight that (and suggest
// that they borrow the RHS mutably).
//
// This implementation attempts to emulate AST-borrowck prioritization
// by trying (3.), then (2.) and finally falling back on (1.).
fn suggest_ampmut<'cx, 'gcx, 'tcx>(
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    mir: &Mir<'tcx>,
    local: Local,
    local_decl: &mir::LocalDecl<'tcx>,
    opt_ty_info: Option<Span>,
) -> (Span, String) {
    let locations = mir.find_assignments(local);
    if locations.len() > 0 {
        let assignment_rhs_span = mir.source_info(locations[0]).span;
        let snippet = tcx.sess.codemap().span_to_snippet(assignment_rhs_span);
        if let Ok(src) = snippet {
            if src.starts_with('&') {
                let borrowed_expr = src[1..].to_string();
                return (
                    assignment_rhs_span,
                    format!("&mut {}", borrowed_expr),
                );
            }
        }
    }

    let highlight_span = match opt_ty_info {
        // if this is a variable binding with an explicit type,
        // try to highlight that for the suggestion.
        Some(ty_span) => ty_span,

        // otherwise, just highlight the span associated with
        // the (MIR) LocalDecl.
        None => local_decl.source_info.span,
    };

    let ty_mut = local_decl.ty.builtin_deref(true).unwrap();
    assert_eq!(ty_mut.mutbl, hir::MutImmutable);
    (highlight_span, format!("&mut {}", ty_mut.ty))
}
