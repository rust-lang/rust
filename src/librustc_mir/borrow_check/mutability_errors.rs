use rustc::hir;
use rustc::hir::Node;
use rustc::mir::{self, BindingForm, Constant, ClearCrossCrate, Local, Location, Body};
use rustc::mir::{
    Mutability, Operand, Place, PlaceBase, Projection, ProjectionElem, Static, StaticKind,
};
use rustc::mir::{Terminator, TerminatorKind};
use rustc::ty::{self, Const, DefIdTree, Ty, TyS, TyCtxt};
use rustc_data_structures::indexed_vec::Idx;
use syntax_pos::Span;
use syntax_pos::symbol::kw;

use crate::dataflow::move_paths::InitLocation;
use crate::borrow_check::MirBorrowckCtxt;
use crate::util::borrowck_errors::{BorrowckErrors, Origin};
use crate::util::collect_writes::FindAssignments;
use crate::util::suggest_ref_mut;
use rustc_errors::Applicability;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum AccessKind {
    MutableBorrow,
    Mutate,
    Move,
}

impl<'a, 'tcx> MirBorrowckCtxt<'a, 'tcx> {
    pub(super) fn report_mutability_error(
        &mut self,
        access_place: &Place<'tcx>,
        span: Span,
        the_place_err: &Place<'tcx>,
        error_access: AccessKind,
        location: Location,
    ) {
        debug!(
            "report_mutability_error(\
                access_place={:?}, span={:?}, the_place_err={:?}, error_access={:?}, location={:?},\
            )",
            access_place, span, the_place_err, error_access, location,
        );

        let mut err;
        let item_msg;
        let reason;
        let access_place_desc = self.describe_place(access_place);
        debug!("report_mutability_error: access_place_desc={:?}", access_place_desc);

        match the_place_err {
            Place::Base(PlaceBase::Local(local)) => {
                item_msg = format!("`{}`", access_place_desc.unwrap());
                if let Place::Base(PlaceBase::Local(_)) = access_place {
                    reason = ", as it is not declared as mutable".to_string();
                } else {
                    let name = self.body.local_decls[*local]
                        .name
                        .expect("immutable unnamed local");
                    reason = format!(", as `{}` is not declared as mutable", name);
                }
            }

            Place::Projection(box Projection {
                base,
                elem: ProjectionElem::Field(upvar_index, _),
            }) => {
                debug_assert!(is_closure_or_generator(
                    base.ty(self.body, self.infcx.tcx).ty
                ));

                item_msg = format!("`{}`", access_place_desc.unwrap());
                if self.is_upvar_field_projection(access_place).is_some() {
                    reason = ", as it is not declared as mutable".to_string();
                } else {
                    let name = self.upvars[upvar_index.index()].name;
                    reason = format!(", as `{}` is not declared as mutable", name);
                }
            }

            Place::Projection(box Projection {
                base,
                elem: ProjectionElem::Deref,
            }) => {
                if *base == Place::Base(PlaceBase::Local(Local::new(1))) &&
                    !self.upvars.is_empty() {
                    item_msg = format!("`{}`", access_place_desc.unwrap());
                    debug_assert!(self.body.local_decls[Local::new(1)].ty.is_region_ptr());
                    debug_assert!(is_closure_or_generator(
                        the_place_err.ty(self.body, self.infcx.tcx).ty
                    ));

                    reason = if self.is_upvar_field_projection(access_place).is_some() {
                        ", as it is a captured variable in a `Fn` closure".to_string()
                    } else {
                        ", as `Fn` closures cannot mutate their captured variables".to_string()
                    }
                } else if {
                    if let Place::Base(PlaceBase::Local(local)) = *base {
                        self.body.local_decls[local].is_ref_for_guard()
                    } else {
                        false
                    }
                } {
                    item_msg = format!("`{}`", access_place_desc.unwrap());
                    reason = ", as it is immutable for the pattern guard".to_string();
                } else {
                    let pointer_type =
                        if base.ty(self.body, self.infcx.tcx).ty.is_region_ptr() {
                            "`&` reference"
                        } else {
                            "`*const` pointer"
                        };
                    if let Some(desc) = access_place_desc {
                        item_msg = format!("`{}`", desc);
                        reason = match error_access {
                            AccessKind::Move |
                            AccessKind::Mutate => format!(" which is behind a {}", pointer_type),
                            AccessKind::MutableBorrow => {
                                format!(", as it is behind a {}", pointer_type)
                            }
                        }
                    } else {
                        item_msg = format!("data in a {}", pointer_type);
                        reason = String::new();
                    }
                }
            }

            Place::Base(PlaceBase::Static(box Static { kind: StaticKind::Promoted(_), .. })) =>
                unreachable!(),

            Place::Base(PlaceBase::Static(box Static { kind: StaticKind::Static(def_id), .. })) => {
                if let Place::Base(PlaceBase::Static(_)) = access_place {
                    item_msg = format!("immutable static item `{}`", access_place_desc.unwrap());
                    reason = String::new();
                } else {
                    item_msg = format!("`{}`", access_place_desc.unwrap());
                    let static_name = &self.infcx.tcx.item_name(*def_id);
                    reason = format!(", as `{}` is an immutable static item", static_name);
                }
            }

            Place::Projection(box Projection {
                base: _,
                elem: ProjectionElem::Index(_),
            })
            | Place::Projection(box Projection {
                base: _,
                elem: ProjectionElem::ConstantIndex { .. },
            })
            | Place::Projection(box Projection {
                base: _,
                elem: ProjectionElem::Subslice { .. },
            })
            | Place::Projection(box Projection {
                base: _,
                elem: ProjectionElem::Downcast(..),
            }) => bug!("Unexpected immutable place."),
        }

        debug!("report_mutability_error: item_msg={:?}, reason={:?}", item_msg, reason);

        // `act` and `acted_on` are strings that let us abstract over
        // the verbs used in some diagnostic messages.
        let act;
        let acted_on;

        let span = match error_access {
            AccessKind::Move => {
                err = self.infcx.tcx
                    .cannot_move_out_of(span, &(item_msg + &reason), Origin::Mir);
                err.span_label(span, "cannot move");
                err.buffer(&mut self.errors_buffer);
                return;
            }
            AccessKind::Mutate => {
                err = self.infcx.tcx
                    .cannot_assign(span, &(item_msg + &reason), Origin::Mir);
                act = "assign";
                acted_on = "written";
                span
            }
            AccessKind::MutableBorrow => {
                act = "borrow as mutable";
                acted_on = "borrowed as mutable";

                let borrow_spans = self.borrow_spans(span, location);
                let borrow_span = borrow_spans.args_or_use();
                err = self.infcx.tcx.cannot_borrow_path_as_mutable_because(
                    borrow_span,
                    &item_msg,
                    &reason,
                    Origin::Mir,
                );
                borrow_spans.var_span_label(
                    &mut err,
                    format!(
                        "mutable borrow occurs due to use of `{}` in closure",
                        // always Some() if the message is printed.
                        self.describe_place(access_place).unwrap_or_default(),
                    )
                );
                borrow_span
            }
        };

        debug!("report_mutability_error: act={:?}, acted_on={:?}", act, acted_on);

        match the_place_err {
            // Suggest making an existing shared borrow in a struct definition a mutable borrow.
            //
            // This is applicable when we have a deref of a field access to a deref of a local -
            // something like `*((*_1).0`. The local that we get will be a reference to the
            // struct we've got a field access of (it must be a reference since there's a deref
            // after the field access).
            Place::Projection(box Projection {
                base: Place::Projection(box Projection {
                    base: Place::Projection(box Projection {
                        base,
                        elem: ProjectionElem::Deref,
                    }),
                    elem: ProjectionElem::Field(field, _),
                }),
                elem: ProjectionElem::Deref,
            }) => {
                err.span_label(span, format!("cannot {ACT}", ACT = act));

                if let Some((span, message)) = annotate_struct_field(
                    self.infcx.tcx,
                    base.ty(self.body, self.infcx.tcx).ty,
                    field,
                ) {
                    err.span_suggestion(
                        span,
                        "consider changing this to be mutable",
                        message,
                        Applicability::MaybeIncorrect,
                    );
                }
            },

            // Suggest removing a `&mut` from the use of a mutable reference.
            Place::Base(PlaceBase::Local(local))
                if {
                    self.body.local_decls.get(*local).map(|local_decl| {
                        if let ClearCrossCrate::Set(
                            mir::BindingForm::ImplicitSelf(kind)
                        ) = local_decl.is_user_variable.as_ref().unwrap() {
                            // Check if the user variable is a `&mut self` and we can therefore
                            // suggest removing the `&mut`.
                            //
                            // Deliberately fall into this case for all implicit self types,
                            // so that we don't fall in to the next case with them.
                            *kind == mir::ImplicitSelfKind::MutRef
                        } else if Some(kw::SelfLower) == local_decl.name {
                            // Otherwise, check if the name is the self kewyord - in which case
                            // we have an explicit self. Do the same thing in this case and check
                            // for a `self: &mut Self` to suggest removing the `&mut`.
                            if let ty::Ref(
                                _, _, hir::Mutability::MutMutable
                            ) = local_decl.ty.sty {
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }).unwrap_or(false)
                } =>
            {
                err.span_label(span, format!("cannot {ACT}", ACT = act));
                err.span_label(span, "try removing `&mut` here");
            },

            // We want to suggest users use `let mut` for local (user
            // variable) mutations...
            Place::Base(PlaceBase::Local(local))
                if self.body.local_decls[*local].can_be_made_mutable() => {
                // ... but it doesn't make sense to suggest it on
                // variables that are `ref x`, `ref mut x`, `&self`,
                // or `&mut self` (such variables are simply not
                // mutable).
                let local_decl = &self.body.local_decls[*local];
                assert_eq!(local_decl.mutability, Mutability::Not);

                err.span_label(span, format!("cannot {ACT}", ACT = act));
                err.span_suggestion(
                    local_decl.source_info.span,
                    "consider changing this to be mutable",
                    format!("mut {}", local_decl.name.unwrap()),
                    Applicability::MachineApplicable,
                );
            }

            // Also suggest adding mut for upvars
            Place::Projection(box Projection {
                base,
                elem: ProjectionElem::Field(upvar_index, _),
            }) => {
                debug_assert!(is_closure_or_generator(
                    base.ty(self.body, self.infcx.tcx).ty
                ));

                err.span_label(span, format!("cannot {ACT}", ACT = act));

                let upvar_hir_id = self.upvars[upvar_index.index()].var_hir_id;
                if let Some(Node::Binding(pat)) = self.infcx.tcx.hir().find(upvar_hir_id)
                {
                    if let hir::PatKind::Binding(
                        hir::BindingAnnotation::Unannotated,
                        _,
                        upvar_ident,
                        _,
                    ) = pat.node
                    {
                        err.span_suggestion(
                            upvar_ident.span,
                            "consider changing this to be mutable",
                            format!("mut {}", upvar_ident.name),
                            Applicability::MachineApplicable,
                        );
                    }
                }
            }

            // complete hack to approximate old AST-borrowck
            // diagnostic: if the span starts with a mutable borrow of
            // a local variable, then just suggest the user remove it.
            Place::Base(PlaceBase::Local(_))
                if {
                    if let Ok(snippet) = self.infcx.tcx.sess.source_map().span_to_snippet(span) {
                        snippet.starts_with("&mut ")
                    } else {
                        false
                    }
                } =>
            {
                err.span_label(span, format!("cannot {ACT}", ACT = act));
                err.span_label(span, "try removing `&mut` here");
            }

            Place::Projection(box Projection {
                base: Place::Base(PlaceBase::Local(local)),
                elem: ProjectionElem::Deref,
            }) if {
                if let Some(ClearCrossCrate::Set(BindingForm::RefForGuard)) =
                    self.body.local_decls[*local].is_user_variable
                {
                    true
                } else {
                    false
                }
            } =>
            {
                err.span_label(span, format!("cannot {ACT}", ACT = act));
                err.note(
                    "variables bound in patterns are immutable until the end of the pattern guard",
                );
            }

            // We want to point out when a `&` can be readily replaced
            // with an `&mut`.
            //
            // FIXME: can this case be generalized to work for an
            // arbitrary base for the projection?
            Place::Projection(box Projection {
                base: Place::Base(PlaceBase::Local(local)),
                elem: ProjectionElem::Deref,
            }) if self.body.local_decls[*local].is_user_variable.is_some() =>
            {
                let local_decl = &self.body.local_decls[*local];
                let suggestion = match local_decl.is_user_variable.as_ref().unwrap() {
                    ClearCrossCrate::Set(mir::BindingForm::ImplicitSelf(_)) => {
                        Some(suggest_ampmut_self(self.infcx.tcx, local_decl))
                    }

                    ClearCrossCrate::Set(mir::BindingForm::Var(mir::VarBindingForm {
                        binding_mode: ty::BindingMode::BindByValue(_),
                        opt_ty_info,
                        ..
                    })) => Some(suggest_ampmut(
                        self.infcx.tcx,
                        self.body,
                        *local,
                        local_decl,
                        *opt_ty_info,
                    )),

                    ClearCrossCrate::Set(mir::BindingForm::Var(mir::VarBindingForm {
                        binding_mode: ty::BindingMode::BindByReference(_),
                        ..
                    })) => {
                        let pattern_span = local_decl.source_info.span;
                        suggest_ref_mut(self.infcx.tcx, pattern_span)
                            .map(|replacement| (pattern_span, replacement))
                    }

                    ClearCrossCrate::Set(mir::BindingForm::RefForGuard) => unreachable!(),

                    ClearCrossCrate::Clear => bug!("saw cleared local state"),
                };

                let (pointer_sigil, pointer_desc) = if local_decl.ty.is_region_ptr() {
                    ("&", "reference")
                } else {
                    ("*const", "pointer")
                };

                if let Some((err_help_span, suggested_code)) = suggestion {
                    err.span_suggestion(
                        err_help_span,
                        &format!("consider changing this to be a mutable {}", pointer_desc),
                        suggested_code,
                        Applicability::MachineApplicable,
                    );
                }

                match local_decl.name {
                    Some(name) if !local_decl.from_compiler_desugaring() => {
                        err.span_label(
                            span,
                            format!(
                                "`{NAME}` is a `{SIGIL}` {DESC}, \
                                so the data it refers to cannot be {ACTED_ON}",
                                NAME = name,
                                SIGIL = pointer_sigil,
                                DESC = pointer_desc,
                                ACTED_ON = acted_on
                            ),
                        );
                    }
                    _ => {
                        err.span_label(
                            span,
                            format!(
                                "cannot {ACT} through `{SIGIL}` {DESC}",
                                ACT = act,
                                SIGIL = pointer_sigil,
                                DESC = pointer_desc
                            ),
                        );
                    }
                }
            }

            Place::Projection(box Projection {
                base,
                elem: ProjectionElem::Deref,
            }) if *base == Place::Base(PlaceBase::Local(Local::new(1))) &&
                  !self.upvars.is_empty() =>
            {
                err.span_label(span, format!("cannot {ACT}", ACT = act));
                err.span_help(
                    self.body.span,
                    "consider changing this to accept closures that implement `FnMut`"
                );
            }

            Place::Projection(box Projection {
                base: Place::Base(PlaceBase::Local(local)),
                elem: ProjectionElem::Deref,
            })  if error_access == AccessKind::MutableBorrow => {
                err.span_label(span, format!("cannot {ACT}", ACT = act));

                let mpi = self.move_data.rev_lookup.find_local(*local);
                for i in self.move_data.init_path_map[mpi].iter() {
                    if let InitLocation::Statement(location) = self.move_data.inits[*i].location {
                        if let Some(
                            Terminator {
                                kind: TerminatorKind::Call {
                                    func: Operand::Constant(box Constant {
                                        literal: Const {
                                            ty: &TyS {
                                                sty: ty::FnDef(id, substs),
                                                ..
                                            },
                                            ..
                                        },
                                        ..
                                    }),
                                    ..
                                },
                                ..
                            }
                        ) = &self.body.basic_blocks()[location.block].terminator {
                            let index_trait = self.infcx.tcx.lang_items().index_trait();
                            if self.infcx.tcx.parent(id) == index_trait {
                                let mut found = false;
                                self.infcx.tcx.for_each_relevant_impl(
                                    self.infcx.tcx.lang_items().index_mut_trait().unwrap(),
                                    substs.type_at(0),
                                    |_relevant_impl| {
                                        found = true;
                                    }
                                );

                                let extra = if found {
                                    String::new()
                                } else {
                                    format!(", but it is not implemented for `{}`",
                                            substs.type_at(0))
                                };

                                err.help(
                                    &format!(
                                        "trait `IndexMut` is required to modify indexed content{}",
                                         extra,
                                    ),
                                );
                            }
                        }
                    }
                }
            }

            _ => {
                err.span_label(span, format!("cannot {ACT}", ACT = act));
            }
        }

        err.buffer(&mut self.errors_buffer);
    }
}

fn suggest_ampmut_self<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_decl: &mir::LocalDecl<'tcx>,
) -> (Span, String) {
    let sp = local_decl.source_info.span;
    (sp, match tcx.sess.source_map().span_to_snippet(sp) {
        Ok(snippet) => {
            let lt_pos = snippet.find('\'');
            if let Some(lt_pos) = lt_pos {
                format!("&{}mut self", &snippet[lt_pos..snippet.len() - 4])
            } else {
                "&mut self".to_string()
            }
        }
        _ => "&mut self".to_string()
    })
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
fn suggest_ampmut<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    local: Local,
    local_decl: &mir::LocalDecl<'tcx>,
    opt_ty_info: Option<Span>,
) -> (Span, String) {
    let locations = body.find_assignments(local);
    if !locations.is_empty() {
        let assignment_rhs_span = body.source_info(locations[0]).span;
        if let Ok(src) = tcx.sess.source_map().span_to_snippet(assignment_rhs_span) {
            if let (true, Some(ws_pos)) = (
                src.starts_with("&'"),
                src.find(|c: char| -> bool { c.is_whitespace() }),
            ) {
                let lt_name = &src[1..ws_pos];
                let ty = &src[ws_pos..];
                return (assignment_rhs_span, format!("&{} mut {}", lt_name, ty));
            } else if src.starts_with('&') {
                let borrowed_expr = &src[1..];
                return (assignment_rhs_span, format!("&mut {}", borrowed_expr));
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

    if let Ok(src) = tcx.sess.source_map().span_to_snippet(highlight_span) {
        if let (true, Some(ws_pos)) = (
            src.starts_with("&'"),
            src.find(|c: char| -> bool { c.is_whitespace() }),
        ) {
            let lt_name = &src[1..ws_pos];
            let ty = &src[ws_pos..];
            return (highlight_span, format!("&{} mut{}", lt_name, ty));
        }
    }

    let ty_mut = local_decl.ty.builtin_deref(true).unwrap();
    assert_eq!(ty_mut.mutbl, hir::MutImmutable);
    (highlight_span,
     if local_decl.ty.is_region_ptr() {
         format!("&mut {}", ty_mut.ty)
     } else {
         format!("*mut {}", ty_mut.ty)
     })
}

fn is_closure_or_generator(ty: Ty<'_>) -> bool {
    ty.is_closure() || ty.is_generator()
}

/// Adds a suggestion to a struct definition given a field access to a local.
/// This function expects the local to be a reference to a struct in order to produce a suggestion.
///
/// ```text
/// LL |     s: &'a String
///    |        ---------- use `&'a mut String` here to make mutable
/// ```
fn annotate_struct_field(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    field: &mir::Field,
) -> Option<(Span, String)> {
    // Expect our local to be a reference to a struct of some kind.
    if let ty::Ref(_, ty, _) = ty.sty {
        if let ty::Adt(def, _) = ty.sty {
            let field = def.all_fields().nth(field.index())?;
            // Use the HIR types to construct the diagnostic message.
            let hir_id = tcx.hir().as_local_hir_id(field.did)?;
            let node = tcx.hir().find(hir_id)?;
            // Now we're dealing with the actual struct that we're going to suggest a change to,
            // we can expect a field that is an immutable reference to a type.
            if let hir::Node::Field(field) = node {
                if let hir::TyKind::Rptr(lifetime, hir::MutTy {
                    mutbl: hir::Mutability::MutImmutable,
                    ref ty
                }) = field.ty.node {
                    // Get the snippets in two parts - the named lifetime (if there is one) and
                    // type being referenced, that way we can reconstruct the snippet without loss
                    // of detail.
                    let type_snippet = tcx.sess.source_map().span_to_snippet(ty.span).ok()?;
                    let lifetime_snippet = if !lifetime.is_elided() {
                        format!("{} ", tcx.sess.source_map().span_to_snippet(lifetime.span).ok()?)
                    } else {
                        String::new()
                    };

                    return Some((
                        field.ty.span,
                        format!(
                            "&{}mut {}",
                            lifetime_snippet, &*type_snippet,
                        ),
                    ));
                }
            }
        }
    }

    None
}
