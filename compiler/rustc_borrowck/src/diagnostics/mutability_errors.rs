#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use core::ops::ControlFlow;

use either::Either;
use hir::{ExprKind, Param};
use rustc_abi::FieldIdx;
use rustc_errors::{Applicability, Diag};
use rustc_hir::intravisit::Visitor;
use rustc_hir::{self as hir, BindingMode, ByRef, Node};
use rustc_middle::bug;
use rustc_middle::hir::place::PlaceBase;
use rustc_middle::mir::visit::PlaceContext;
use rustc_middle::mir::{
    self, BindingForm, Body, BorrowKind, Local, LocalDecl, LocalInfo, LocalKind, Location,
    Mutability, Operand, Place, PlaceRef, ProjectionElem, RawPtrKind, Rvalue, Statement,
    StatementKind, TerminatorKind,
};
use rustc_middle::ty::{self, InstanceKind, Ty, TyCtxt, Upcast};
use rustc_span::{BytePos, DesugaringKind, Span, Symbol, kw, sym};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits;
use tracing::{debug, trace};

use crate::diagnostics::BorrowedContentSource;
use crate::{MirBorrowckCtxt, session_diagnostics};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum AccessKind {
    MutableBorrow,
    Mutate,
}

/// Finds all statements that assign directly to local (i.e., X = ...) and returns their
/// locations.
fn find_assignments(body: &Body<'_>, local: Local) -> Vec<Location> {
    use rustc_middle::mir::visit::Visitor;

    struct FindLocalAssignmentVisitor {
        needle: Local,
        locations: Vec<Location>,
    }

    impl<'tcx> Visitor<'tcx> for FindLocalAssignmentVisitor {
        fn visit_local(&mut self, local: Local, place_context: PlaceContext, location: Location) {
            if self.needle != local {
                return;
            }

            if place_context.is_place_assignment() {
                self.locations.push(location);
            }
        }
    }

    let mut visitor = FindLocalAssignmentVisitor { needle: local, locations: vec![] };
    visitor.visit_body(body);
    visitor.locations
}

impl<'infcx, 'tcx> MirBorrowckCtxt<'_, 'infcx, 'tcx> {
    pub(crate) fn report_mutability_error(
        &mut self,
        access_place: Place<'tcx>,
        span: Span,
        the_place_err: PlaceRef<'tcx>,
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
        let mut opt_source = None;
        let access_place_desc = self.describe_any_place(access_place.as_ref());
        debug!("report_mutability_error: access_place_desc={:?}", access_place_desc);

        match the_place_err {
            PlaceRef { local, projection: [] } => {
                item_msg = access_place_desc;
                if access_place.as_local().is_some() {
                    reason = ", as it is not declared as mutable".to_string();
                } else {
                    let name = self.local_name(local).expect("immutable unnamed local");
                    reason = format!(", as `{name}` is not declared as mutable");
                }
            }

            PlaceRef {
                local,
                projection: [proj_base @ .., ProjectionElem::Field(upvar_index, _)],
            } => {
                debug_assert!(is_closure_like(
                    Place::ty_from(local, proj_base, self.body, self.infcx.tcx).ty
                ));

                let imm_borrow_derefed = self.upvars[upvar_index.index()]
                    .place
                    .deref_tys()
                    .any(|ty| matches!(ty.kind(), ty::Ref(.., hir::Mutability::Not)));

                // If the place is immutable then:
                //
                // - Either we deref an immutable ref to get to our final place.
                //    - We don't capture derefs of raw ptrs
                // - Or the final place is immut because the root variable of the capture
                //   isn't marked mut and we should suggest that to the user.
                if imm_borrow_derefed {
                    // If we deref an immutable ref then the suggestion here doesn't help.
                    return;
                } else {
                    item_msg = access_place_desc;
                    if self.is_upvar_field_projection(access_place.as_ref()).is_some() {
                        reason = ", as it is not declared as mutable".to_string();
                    } else {
                        let name = self.upvars[upvar_index.index()].to_string(self.infcx.tcx);
                        reason = format!(", as `{name}` is not declared as mutable");
                    }
                }
            }

            PlaceRef { local, projection: [ProjectionElem::Deref] }
                if self.body.local_decls[local].is_ref_for_guard() =>
            {
                item_msg = access_place_desc;
                reason = ", as it is immutable for the pattern guard".to_string();
            }
            PlaceRef { local, projection: [ProjectionElem::Deref] }
                if self.body.local_decls[local].is_ref_to_static() =>
            {
                if access_place.projection.len() == 1 {
                    item_msg = format!("immutable static item {access_place_desc}");
                    reason = String::new();
                } else {
                    item_msg = access_place_desc;
                    let local_info = self.body.local_decls[local].local_info();
                    if let LocalInfo::StaticRef { def_id, .. } = *local_info {
                        let static_name = &self.infcx.tcx.item_name(def_id);
                        reason = format!(", as `{static_name}` is an immutable static item");
                    } else {
                        bug!("is_ref_to_static return true, but not ref to static?");
                    }
                }
            }
            PlaceRef { local: _, projection: [proj_base @ .., ProjectionElem::Deref] } => {
                if the_place_err.local == ty::CAPTURE_STRUCT_LOCAL
                    && proj_base.is_empty()
                    && !self.upvars.is_empty()
                {
                    item_msg = access_place_desc;
                    debug_assert!(self.body.local_decls[ty::CAPTURE_STRUCT_LOCAL].ty.is_ref());
                    debug_assert!(is_closure_like(the_place_err.ty(self.body, self.infcx.tcx).ty));

                    reason = if self.is_upvar_field_projection(access_place.as_ref()).is_some() {
                        ", as it is a captured variable in a `Fn` closure".to_string()
                    } else {
                        ", as `Fn` closures cannot mutate their captured variables".to_string()
                    }
                } else {
                    let source = self.borrowed_content_source(PlaceRef {
                        local: the_place_err.local,
                        projection: proj_base,
                    });
                    let pointer_type = source.describe_for_immutable_place(self.infcx.tcx);
                    opt_source = Some(source);
                    if let Some(desc) = self.describe_place(access_place.as_ref()) {
                        item_msg = format!("`{desc}`");
                        reason = match error_access {
                            AccessKind::Mutate => format!(", which is behind {pointer_type}"),
                            AccessKind::MutableBorrow => {
                                format!(", as it is behind {pointer_type}")
                            }
                        }
                    } else {
                        item_msg = format!("data in {pointer_type}");
                        reason = String::new();
                    }
                }
            }

            PlaceRef {
                local: _,
                projection:
                    [
                        ..,
                        ProjectionElem::Index(_)
                        | ProjectionElem::ConstantIndex { .. }
                        | ProjectionElem::OpaqueCast { .. }
                        | ProjectionElem::Subslice { .. }
                        | ProjectionElem::Downcast(..)
                        | ProjectionElem::UnwrapUnsafeBinder(_),
                    ],
            } => bug!("Unexpected immutable place."),
        }

        debug!("report_mutability_error: item_msg={:?}, reason={:?}", item_msg, reason);

        // `act` and `acted_on` are strings that let us abstract over
        // the verbs used in some diagnostic messages.
        let act;
        let acted_on;
        let mut suggest = true;
        let mut mut_error = None;
        let mut count = 1;

        let span = match error_access {
            AccessKind::Mutate => {
                err = self.cannot_assign(span, &(item_msg + &reason));
                act = "assign";
                acted_on = "written";
                span
            }
            AccessKind::MutableBorrow => {
                act = "borrow as mutable";
                acted_on = "borrowed as mutable";

                let borrow_spans = self.borrow_spans(span, location);
                let borrow_span = borrow_spans.args_or_use();
                match the_place_err {
                    PlaceRef { local, projection: [] }
                        if self.body.local_decls[local].can_be_made_mutable() =>
                    {
                        let span = self.body.local_decls[local].source_info.span;
                        mut_error = Some(span);
                        if let Some((buffered_err, c)) = self.get_buffered_mut_error(span) {
                            // We've encountered a second (or more) attempt to mutably borrow an
                            // immutable binding, so the likely problem is with the binding
                            // declaration, not the use. We collect these in a single diagnostic
                            // and make the binding the primary span of the error.
                            err = buffered_err;
                            count = c + 1;
                            if count == 2 {
                                err.replace_span_with(span, false);
                                err.span_label(span, "not mutable");
                            }
                            suggest = false;
                        } else {
                            err = self.cannot_borrow_path_as_mutable_because(
                                borrow_span,
                                &item_msg,
                                &reason,
                            );
                        }
                    }
                    _ => {
                        err = self.cannot_borrow_path_as_mutable_because(
                            borrow_span,
                            &item_msg,
                            &reason,
                        );
                    }
                }
                if suggest {
                    borrow_spans.var_subdiag(
                        &mut err,
                        Some(mir::BorrowKind::Mut { kind: mir::MutBorrowKind::Default }),
                        |_kind, var_span| {
                            let place = self.describe_any_place(access_place.as_ref());
                            session_diagnostics::CaptureVarCause::MutableBorrowUsePlaceClosure {
                                place,
                                var_span,
                            }
                        },
                    );
                }
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
            PlaceRef {
                local,
                projection:
                    [
                        proj_base @ ..,
                        ProjectionElem::Deref,
                        ProjectionElem::Field(field, _),
                        ProjectionElem::Deref,
                    ],
            } => {
                err.span_label(span, format!("cannot {act}"));

                let place = Place::ty_from(local, proj_base, self.body, self.infcx.tcx);
                if let Some(span) = get_mut_span_in_struct_field(self.infcx.tcx, place.ty, *field) {
                    err.span_suggestion_verbose(
                        span,
                        "consider changing this to be mutable",
                        " mut ",
                        Applicability::MaybeIncorrect,
                    );
                }
            }

            // Suggest removing a `&mut` from the use of a mutable reference.
            PlaceRef { local, projection: [] }
                if self
                    .body
                    .local_decls
                    .get(local)
                    .is_some_and(|l| mut_borrow_of_mutable_ref(l, self.local_name(local))) =>
            {
                let decl = &self.body.local_decls[local];
                err.span_label(span, format!("cannot {act}"));
                if let Some(mir::Statement {
                    source_info,
                    kind:
                        mir::StatementKind::Assign(box (
                            _,
                            mir::Rvalue::Ref(
                                _,
                                mir::BorrowKind::Mut { kind: mir::MutBorrowKind::Default },
                                _,
                            ),
                        )),
                    ..
                }) = &self.body[location.block].statements.get(location.statement_index)
                {
                    match *decl.local_info() {
                        LocalInfo::User(BindingForm::Var(mir::VarBindingForm {
                            binding_mode: BindingMode(ByRef::No, Mutability::Not),
                            opt_ty_info: Some(sp),
                            opt_match_place: _,
                            pat_span: _,
                        })) => {
                            if suggest {
                                err.span_note(sp, "the binding is already a mutable borrow");
                            }
                        }
                        _ => {
                            err.span_note(
                                decl.source_info.span,
                                "the binding is already a mutable borrow",
                            );
                        }
                    }
                    if let Ok(snippet) =
                        self.infcx.tcx.sess.source_map().span_to_snippet(source_info.span)
                    {
                        if snippet.starts_with("&mut ") {
                            // We don't have access to the HIR to get accurate spans, but we can
                            // give a best effort structured suggestion.
                            err.span_suggestion_verbose(
                                source_info.span.with_hi(source_info.span.lo() + BytePos(5)),
                                "try removing `&mut` here",
                                "",
                                Applicability::MachineApplicable,
                            );
                        } else {
                            // This can occur with things like `(&mut self).foo()`.
                            err.span_help(source_info.span, "try removing `&mut` here");
                        }
                    } else {
                        err.span_help(source_info.span, "try removing `&mut` here");
                    }
                } else if decl.mutability.is_not() {
                    if matches!(
                        decl.local_info(),
                        LocalInfo::User(BindingForm::ImplicitSelf(hir::ImplicitSelfKind::RefMut))
                    ) {
                        err.note(
                            "as `Self` may be unsized, this call attempts to take `&mut &mut self`",
                        );
                        err.note("however, `&mut self` expands to `self: &mut Self`, therefore `self` cannot be borrowed mutably");
                    } else {
                        err.span_suggestion_verbose(
                            decl.source_info.span.shrink_to_lo(),
                            "consider making the binding mutable",
                            "mut ",
                            Applicability::MachineApplicable,
                        );
                    };
                }
            }

            // We want to suggest users use `let mut` for local (user
            // variable) mutations...
            PlaceRef { local, projection: [] }
                if self.body.local_decls[local].can_be_made_mutable() =>
            {
                // ... but it doesn't make sense to suggest it on
                // variables that are `ref x`, `ref mut x`, `&self`,
                // or `&mut self` (such variables are simply not
                // mutable).
                let local_decl = &self.body.local_decls[local];
                assert_eq!(local_decl.mutability, Mutability::Not);

                if count < 10 {
                    err.span_label(span, format!("cannot {act}"));
                }
                if suggest {
                    self.construct_mut_suggestion_for_local_binding_patterns(&mut err, local);
                    let tcx = self.infcx.tcx;
                    if let ty::Closure(id, _) = *the_place_err.ty(self.body, tcx).ty.kind() {
                        self.show_mutating_upvar(tcx, id.expect_local(), the_place_err, &mut err);
                    }
                }
            }

            // Also suggest adding mut for upvars.
            PlaceRef {
                local,
                projection: [proj_base @ .., ProjectionElem::Field(upvar_index, _)],
            } => {
                debug_assert!(is_closure_like(
                    Place::ty_from(local, proj_base, self.body, self.infcx.tcx).ty
                ));

                let captured_place = self.upvars[upvar_index.index()];

                err.span_label(span, format!("cannot {act}"));

                let upvar_hir_id = captured_place.get_root_variable();

                if let Node::Pat(pat) = self.infcx.tcx.hir_node(upvar_hir_id)
                    && let hir::PatKind::Binding(hir::BindingMode::NONE, _, upvar_ident, _) =
                        pat.kind
                {
                    if upvar_ident.name == kw::SelfLower {
                        for (_, node) in self.infcx.tcx.hir_parent_iter(upvar_hir_id) {
                            if let Some(fn_decl) = node.fn_decl() {
                                if !matches!(
                                    fn_decl.implicit_self,
                                    hir::ImplicitSelfKind::RefImm | hir::ImplicitSelfKind::RefMut
                                ) {
                                    err.span_suggestion_verbose(
                                        upvar_ident.span.shrink_to_lo(),
                                        "consider changing this to be mutable",
                                        "mut ",
                                        Applicability::MachineApplicable,
                                    );
                                    break;
                                }
                            }
                        }
                    } else {
                        err.span_suggestion_verbose(
                            upvar_ident.span.shrink_to_lo(),
                            "consider changing this to be mutable",
                            "mut ",
                            Applicability::MachineApplicable,
                        );
                    }
                }

                let tcx = self.infcx.tcx;
                if let ty::Ref(_, ty, Mutability::Mut) = the_place_err.ty(self.body, tcx).ty.kind()
                    && let ty::Closure(id, _) = *ty.kind()
                {
                    self.show_mutating_upvar(tcx, id.expect_local(), the_place_err, &mut err);
                }
            }

            // Complete hack to approximate old AST-borrowck diagnostic: if the span starts
            // with a mutable borrow of a local variable, then just suggest the user remove it.
            PlaceRef { local: _, projection: [] }
                if self
                    .infcx
                    .tcx
                    .sess
                    .source_map()
                    .span_to_snippet(span)
                    .is_ok_and(|snippet| snippet.starts_with("&mut ")) =>
            {
                err.span_label(span, format!("cannot {act}"));
                err.span_suggestion_verbose(
                    span.with_hi(span.lo() + BytePos(5)),
                    "try removing `&mut` here",
                    "",
                    Applicability::MaybeIncorrect,
                );
            }

            PlaceRef { local, projection: [ProjectionElem::Deref] }
                if self.body.local_decls[local].is_ref_for_guard() =>
            {
                err.span_label(span, format!("cannot {act}"));
                err.note(
                    "variables bound in patterns are immutable until the end of the pattern guard",
                );
            }

            // We want to point out when a `&` can be readily replaced
            // with an `&mut`.
            //
            // FIXME: can this case be generalized to work for an
            // arbitrary base for the projection?
            PlaceRef { local, projection: [ProjectionElem::Deref] }
                if self.body.local_decls[local].is_user_variable() =>
            {
                let local_decl = &self.body.local_decls[local];

                let (pointer_sigil, pointer_desc) =
                    if local_decl.ty.is_ref() { ("&", "reference") } else { ("*const", "pointer") };

                match self.local_name(local) {
                    Some(name) if !local_decl.from_compiler_desugaring() => {
                        err.span_label(
                            span,
                            format!(
                                "`{name}` is a `{pointer_sigil}` {pointer_desc}, \
                                 so the data it refers to cannot be {acted_on}",
                            ),
                        );

                        self.suggest_using_iter_mut(&mut err);
                        self.suggest_make_local_mut(&mut err, local, name);
                    }
                    _ => {
                        err.span_label(
                            span,
                            format!("cannot {act} through `{pointer_sigil}` {pointer_desc}"),
                        );
                    }
                }
            }

            PlaceRef { local, projection: [ProjectionElem::Deref] }
                if local == ty::CAPTURE_STRUCT_LOCAL && !self.upvars.is_empty() =>
            {
                self.expected_fn_found_fn_mut_call(&mut err, span, act);
            }

            PlaceRef { local: _, projection: [.., ProjectionElem::Deref] } => {
                err.span_label(span, format!("cannot {act}"));

                match opt_source {
                    Some(BorrowedContentSource::OverloadedDeref(ty)) => {
                        err.help(format!(
                            "trait `DerefMut` is required to modify through a dereference, \
                             but it is not implemented for `{ty}`",
                        ));
                    }
                    Some(BorrowedContentSource::OverloadedIndex(ty)) => {
                        err.help(format!(
                            "trait `IndexMut` is required to modify indexed content, \
                             but it is not implemented for `{ty}`",
                        ));
                        self.suggest_map_index_mut_alternatives(ty, &mut err, span);
                    }
                    _ => (),
                }
            }

            _ => {
                err.span_label(span, format!("cannot {act}"));
            }
        }

        if let Some(span) = mut_error {
            self.buffer_mut_error(span, err, count);
        } else {
            self.buffer_error(err);
        }
    }

    /// Suggest `map[k] = v` => `map.insert(k, v)` and the like.
    fn suggest_map_index_mut_alternatives(&self, ty: Ty<'tcx>, err: &mut Diag<'infcx>, span: Span) {
        let Some(adt) = ty.ty_adt_def() else { return };
        let did = adt.did();
        if self.infcx.tcx.is_diagnostic_item(sym::HashMap, did)
            || self.infcx.tcx.is_diagnostic_item(sym::BTreeMap, did)
        {
            /// Walks through the HIR, looking for the corresponding span for this error.
            /// When it finds it, see if it corresponds to assignment operator whose LHS
            /// is an index expr.
            struct SuggestIndexOperatorAlternativeVisitor<'a, 'infcx, 'tcx> {
                assign_span: Span,
                err: &'a mut Diag<'infcx>,
                ty: Ty<'tcx>,
                suggested: bool,
            }
            impl<'a, 'infcx, 'tcx> Visitor<'tcx> for SuggestIndexOperatorAlternativeVisitor<'a, 'infcx, 'tcx> {
                fn visit_stmt(&mut self, stmt: &'tcx hir::Stmt<'tcx>) {
                    hir::intravisit::walk_stmt(self, stmt);
                    let expr = match stmt.kind {
                        hir::StmtKind::Semi(expr) | hir::StmtKind::Expr(expr) => expr,
                        hir::StmtKind::Let(hir::LetStmt { init: Some(expr), .. }) => expr,
                        _ => {
                            return;
                        }
                    };
                    if let hir::ExprKind::Assign(place, rv, _sp) = expr.kind
                        && let hir::ExprKind::Index(val, index, _) = place.kind
                        && (expr.span == self.assign_span || place.span == self.assign_span)
                    {
                        // val[index] = rv;
                        // ---------- place
                        self.err.multipart_suggestions(
                            format!(
                                "use `.insert()` to insert a value into a `{}`, `.get_mut()` \
                                to modify it, or the entry API for more flexibility",
                                self.ty,
                            ),
                            vec![
                                vec![
                                    // val.insert(index, rv);
                                    (
                                        val.span.shrink_to_hi().with_hi(index.span.lo()),
                                        ".insert(".to_string(),
                                    ),
                                    (
                                        index.span.shrink_to_hi().with_hi(rv.span.lo()),
                                        ", ".to_string(),
                                    ),
                                    (rv.span.shrink_to_hi(), ")".to_string()),
                                ],
                                vec![
                                    // if let Some(v) = val.get_mut(index) { *v = rv; }
                                    (val.span.shrink_to_lo(), "if let Some(val) = ".to_string()),
                                    (
                                        val.span.shrink_to_hi().with_hi(index.span.lo()),
                                        ".get_mut(".to_string(),
                                    ),
                                    (
                                        index.span.shrink_to_hi().with_hi(place.span.hi()),
                                        ") { *val".to_string(),
                                    ),
                                    (rv.span.shrink_to_hi(), "; }".to_string()),
                                ],
                                vec![
                                    // let x = val.entry(index).or_insert(rv);
                                    (val.span.shrink_to_lo(), "let val = ".to_string()),
                                    (
                                        val.span.shrink_to_hi().with_hi(index.span.lo()),
                                        ".entry(".to_string(),
                                    ),
                                    (
                                        index.span.shrink_to_hi().with_hi(rv.span.lo()),
                                        ").or_insert(".to_string(),
                                    ),
                                    (rv.span.shrink_to_hi(), ")".to_string()),
                                ],
                            ],
                            Applicability::MachineApplicable,
                        );
                        self.suggested = true;
                    } else if let hir::ExprKind::MethodCall(_path, receiver, _, sp) = expr.kind
                        && let hir::ExprKind::Index(val, index, _) = receiver.kind
                        && receiver.span == self.assign_span
                    {
                        // val[index].path(args..);
                        self.err.multipart_suggestion(
                            format!("to modify a `{}` use `.get_mut()`", self.ty),
                            vec![
                                (val.span.shrink_to_lo(), "if let Some(val) = ".to_string()),
                                (
                                    val.span.shrink_to_hi().with_hi(index.span.lo()),
                                    ".get_mut(".to_string(),
                                ),
                                (
                                    index.span.shrink_to_hi().with_hi(receiver.span.hi()),
                                    ") { val".to_string(),
                                ),
                                (sp.shrink_to_hi(), "; }".to_string()),
                            ],
                            Applicability::MachineApplicable,
                        );
                        self.suggested = true;
                    }
                }
            }
            let def_id = self.body.source.def_id();
            let Some(local_def_id) = def_id.as_local() else { return };
            let Some(body) = self.infcx.tcx.hir_maybe_body_owned_by(local_def_id) else { return };

            let mut v = SuggestIndexOperatorAlternativeVisitor {
                assign_span: span,
                err,
                ty,
                suggested: false,
            };
            v.visit_body(&body);
            if !v.suggested {
                err.help(format!(
                    "to modify a `{ty}`, use `.get_mut()`, `.insert()` or the entry API",
                ));
            }
        }
    }

    /// User cannot make signature of a trait mutable without changing the
    /// trait. So we find if this error belongs to a trait and if so we move
    /// suggestion to the trait or disable it if it is out of scope of this crate
    ///
    /// The returned values are:
    ///  - is the current item an assoc `fn` of an impl that corresponds to a trait def? if so, we
    ///    have to suggest changing both the impl `fn` arg and the trait `fn` arg
    ///  - is the trait from the local crate? If not, we can't suggest changing signatures
    ///  - `Span` of the argument in the trait definition
    fn is_error_in_trait(&self, local: Local) -> (bool, bool, Option<Span>) {
        let tcx = self.infcx.tcx;
        if self.body.local_kind(local) != LocalKind::Arg {
            return (false, false, None);
        }
        let my_def = self.body.source.def_id();
        let Some(td) =
            tcx.trait_impl_of_assoc(my_def).and_then(|id| self.infcx.tcx.trait_id_of_impl(id))
        else {
            return (false, false, None);
        };

        let implemented_trait_item = self.infcx.tcx.trait_item_of(my_def);

        (
            true,
            td.is_local(),
            implemented_trait_item.and_then(|f_in_trait| {
                let f_in_trait = f_in_trait.as_local()?;
                if let Node::TraitItem(ti) = self.infcx.tcx.hir_node_by_def_id(f_in_trait)
                    && let hir::TraitItemKind::Fn(sig, _) = ti.kind
                    && let Some(ty) = sig.decl.inputs.get(local.index() - 1)
                    && let hir::TyKind::Ref(_, mut_ty) = ty.kind
                    && let hir::Mutability::Not = mut_ty.mutbl
                    && sig.decl.implicit_self.has_implicit_self()
                {
                    Some(ty.span)
                } else {
                    None
                }
            }),
        )
    }

    fn construct_mut_suggestion_for_local_binding_patterns(
        &self,
        err: &mut Diag<'_>,
        local: Local,
    ) {
        let local_decl = &self.body.local_decls[local];
        debug!("local_decl: {:?}", local_decl);
        let pat_span = match *local_decl.local_info() {
            LocalInfo::User(BindingForm::Var(mir::VarBindingForm {
                binding_mode: BindingMode(ByRef::No, Mutability::Not),
                opt_ty_info: _,
                opt_match_place: _,
                pat_span,
            })) => pat_span,
            _ => local_decl.source_info.span,
        };

        // With ref-binding patterns, the mutability suggestion has to apply to
        // the binding, not the reference (which would be a type error):
        //
        // `let &b = a;` -> `let &(mut b) = a;`
        // or
        // `fn foo(&x: &i32)` -> `fn foo(&(mut x): &i32)`
        let def_id = self.body.source.def_id();
        if let Some(local_def_id) = def_id.as_local()
            && let Some(body) = self.infcx.tcx.hir_maybe_body_owned_by(local_def_id)
            && let Some(hir_id) = (BindingFinder { span: pat_span }).visit_body(&body).break_value()
            && let node = self.infcx.tcx.hir_node(hir_id)
            && let hir::Node::LetStmt(hir::LetStmt {
                pat: hir::Pat { kind: hir::PatKind::Ref(_, _), .. },
                ..
            })
            | hir::Node::Param(Param {
                pat: hir::Pat { kind: hir::PatKind::Ref(_, _), .. },
                ..
            }) = node
        {
            err.multipart_suggestion(
                "consider changing this to be mutable",
                vec![
                    (pat_span.until(local_decl.source_info.span), "&(mut ".to_string()),
                    (
                        local_decl.source_info.span.shrink_to_hi().with_hi(pat_span.hi()),
                        ")".to_string(),
                    ),
                ],
                Applicability::MachineApplicable,
            );
            return;
        }

        err.span_suggestion_verbose(
            local_decl.source_info.span.shrink_to_lo(),
            "consider changing this to be mutable",
            "mut ",
            Applicability::MachineApplicable,
        );
    }

    // Point to span of upvar making closure call that requires a mutable borrow
    fn show_mutating_upvar(
        &self,
        tcx: TyCtxt<'_>,
        closure_local_def_id: hir::def_id::LocalDefId,
        the_place_err: PlaceRef<'tcx>,
        err: &mut Diag<'_>,
    ) {
        let tables = tcx.typeck(closure_local_def_id);
        if let Some((span, closure_kind_origin)) = tcx.closure_kind_origin(closure_local_def_id) {
            let reason = if let PlaceBase::Upvar(upvar_id) = closure_kind_origin.base {
                let upvar = ty::place_to_string_for_capture(tcx, closure_kind_origin);
                let root_hir_id = upvar_id.var_path.hir_id;
                // We have an origin for this closure kind starting at this root variable so it's
                // safe to unwrap here.
                let captured_places =
                    tables.closure_min_captures[&closure_local_def_id].get(&root_hir_id).unwrap();

                let origin_projection = closure_kind_origin
                    .projections
                    .iter()
                    .map(|proj| proj.kind)
                    .collect::<Vec<_>>();
                let mut capture_reason = String::new();
                for captured_place in captured_places {
                    let captured_place_kinds = captured_place
                        .place
                        .projections
                        .iter()
                        .map(|proj| proj.kind)
                        .collect::<Vec<_>>();
                    if rustc_middle::ty::is_ancestor_or_same_capture(
                        &captured_place_kinds,
                        &origin_projection,
                    ) {
                        match captured_place.info.capture_kind {
                            ty::UpvarCapture::ByRef(
                                ty::BorrowKind::Mutable | ty::BorrowKind::UniqueImmutable,
                            ) => {
                                capture_reason = format!("mutable borrow of `{upvar}`");
                            }
                            ty::UpvarCapture::ByValue | ty::UpvarCapture::ByUse => {
                                capture_reason = format!("possible mutation of `{upvar}`");
                            }
                            _ => bug!("upvar `{upvar}` borrowed, but not mutably"),
                        }
                        break;
                    }
                }
                if capture_reason.is_empty() {
                    bug!("upvar `{upvar}` borrowed, but cannot find reason");
                }
                capture_reason
            } else {
                bug!("not an upvar")
            };
            // Sometimes we deliberately don't store the name of a place when coming from a macro in
            // another crate. We generally want to limit those diagnostics a little, to hide
            // implementation details (such as those from pin!() or format!()). In that case show a
            // slightly different error message, or none at all if something else happened. In other
            // cases the message is likely not useful.
            if let Some(place_name) = self.describe_place(the_place_err) {
                err.span_label(
                    *span,
                    format!("calling `{place_name}` requires mutable binding due to {reason}"),
                );
            } else if span.from_expansion() {
                err.span_label(
                    *span,
                    format!("a call in this macro requires a mutable binding due to {reason}",),
                );
            }
        }
    }

    // Attempt to search similar mutable associated items for suggestion.
    // In the future, attempt in all path but initially for RHS of for_loop
    fn suggest_similar_mut_method_for_for_loop(&self, err: &mut Diag<'_>, span: Span) {
        use hir::ExprKind::{AddrOf, Block, Call, MethodCall};
        use hir::{BorrowKind, Expr};

        let tcx = self.infcx.tcx;
        struct Finder {
            span: Span,
        }

        impl<'tcx> Visitor<'tcx> for Finder {
            type Result = ControlFlow<&'tcx Expr<'tcx>>;
            fn visit_expr(&mut self, e: &'tcx hir::Expr<'tcx>) -> Self::Result {
                if e.span == self.span {
                    ControlFlow::Break(e)
                } else {
                    hir::intravisit::walk_expr(self, e)
                }
            }
        }
        if let Some(body) = tcx.hir_maybe_body_owned_by(self.mir_def_id())
            && let Block(block, _) = body.value.kind
        {
            // `span` corresponds to the expression being iterated, find the `for`-loop desugared
            // expression with that span in order to identify potential fixes when encountering a
            // read-only iterator that should be mutable.
            if let ControlFlow::Break(expr) = (Finder { span }).visit_block(block)
                && let Call(_, [expr]) = expr.kind
            {
                match expr.kind {
                    MethodCall(path_segment, _, _, span) => {
                        // We have `for _ in iter.read_only_iter()`, try to
                        // suggest `for _ in iter.mutable_iter()` instead.
                        let opt_suggestions = tcx
                            .typeck(path_segment.hir_id.owner.def_id)
                            .type_dependent_def_id(expr.hir_id)
                            .and_then(|def_id| tcx.impl_of_assoc(def_id))
                            .map(|def_id| tcx.associated_items(def_id))
                            .map(|assoc_items| {
                                assoc_items
                                    .in_definition_order()
                                    .map(|assoc_item_def| assoc_item_def.ident(tcx))
                                    .filter(|&ident| {
                                        let original_method_ident = path_segment.ident;
                                        original_method_ident != ident
                                            && ident.as_str().starts_with(
                                                &original_method_ident.name.to_string(),
                                            )
                                    })
                                    .map(|ident| format!("{ident}()"))
                                    .peekable()
                            });

                        if let Some(mut suggestions) = opt_suggestions
                            && suggestions.peek().is_some()
                        {
                            err.span_suggestions(
                                span,
                                "use mutable method",
                                suggestions,
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                    AddrOf(BorrowKind::Ref, Mutability::Not, expr) => {
                        // We have `for _ in &i`, suggest `for _ in &mut i`.
                        err.span_suggestion_verbose(
                            expr.span.shrink_to_lo(),
                            "use a mutable iterator instead",
                            "mut ",
                            Applicability::MachineApplicable,
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    /// Targeted error when encountering an `FnMut` closure where an `Fn` closure was expected.
    fn expected_fn_found_fn_mut_call(&self, err: &mut Diag<'_>, sp: Span, act: &str) {
        err.span_label(sp, format!("cannot {act}"));

        let tcx = self.infcx.tcx;
        let closure_id = self.mir_hir_id();
        let closure_span = tcx.def_span(self.mir_def_id());
        let fn_call_id = tcx.parent_hir_id(closure_id);
        let node = tcx.hir_node(fn_call_id);
        let def_id = tcx.hir_enclosing_body_owner(fn_call_id);
        let mut look_at_return = true;

        // If the HIR node is a function or method call, get the DefId
        // of the callee function or method, the span, and args of the call expr
        let get_call_details = || {
            let hir::Node::Expr(hir::Expr { hir_id, kind, .. }) = node else {
                return None;
            };

            let typeck_results = tcx.typeck(def_id);

            match kind {
                hir::ExprKind::Call(expr, args) => {
                    if let Some(ty::FnDef(def_id, _)) =
                        typeck_results.node_type_opt(expr.hir_id).as_ref().map(|ty| ty.kind())
                    {
                        Some((*def_id, expr.span, *args))
                    } else {
                        None
                    }
                }
                hir::ExprKind::MethodCall(_, _, args, span) => typeck_results
                    .type_dependent_def_id(*hir_id)
                    .map(|def_id| (def_id, *span, *args)),
                _ => None,
            }
        };

        // If we can detect the expression to be a function or method call where the closure was
        // an argument, we point at the function or method definition argument...
        if let Some((callee_def_id, call_span, call_args)) = get_call_details() {
            let arg_pos = call_args
                .iter()
                .enumerate()
                .filter(|(_, arg)| arg.hir_id == closure_id)
                .map(|(pos, _)| pos)
                .next();

            let arg = match tcx.hir_get_if_local(callee_def_id) {
                Some(
                    hir::Node::Item(hir::Item {
                        kind: hir::ItemKind::Fn { ident, sig, .. }, ..
                    })
                    | hir::Node::TraitItem(hir::TraitItem {
                        ident,
                        kind: hir::TraitItemKind::Fn(sig, _),
                        ..
                    })
                    | hir::Node::ImplItem(hir::ImplItem {
                        ident,
                        kind: hir::ImplItemKind::Fn(sig, _),
                        ..
                    }),
                ) => Some(
                    arg_pos
                        .and_then(|pos| {
                            sig.decl.inputs.get(
                                pos + if sig.decl.implicit_self.has_implicit_self() {
                                    1
                                } else {
                                    0
                                },
                            )
                        })
                        .map(|arg| arg.span)
                        .unwrap_or(ident.span),
                ),
                _ => None,
            };
            if let Some(span) = arg {
                err.span_label(span, "change this to accept `FnMut` instead of `Fn`");
                err.span_label(call_span, "expects `Fn` instead of `FnMut`");
                err.span_label(closure_span, "in this closure");
                look_at_return = false;
            }
        }

        if look_at_return && tcx.hir_get_fn_id_for_return_block(closure_id).is_some() {
            // ...otherwise we are probably in the tail expression of the function, point at the
            // return type.
            match tcx.hir_node_by_def_id(tcx.hir_get_parent_item(fn_call_id).def_id) {
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Fn { ident, sig, .. }, ..
                })
                | hir::Node::TraitItem(hir::TraitItem {
                    ident,
                    kind: hir::TraitItemKind::Fn(sig, _),
                    ..
                })
                | hir::Node::ImplItem(hir::ImplItem {
                    ident,
                    kind: hir::ImplItemKind::Fn(sig, _),
                    ..
                }) => {
                    err.span_label(ident.span, "");
                    err.span_label(
                        sig.decl.output.span(),
                        "change this to return `FnMut` instead of `Fn`",
                    );
                    err.span_label(closure_span, "in this closure");
                }
                _ => {}
            }
        }
    }

    fn suggest_using_iter_mut(&self, err: &mut Diag<'_>) {
        let source = self.body.source;
        if let InstanceKind::Item(def_id) = source.instance
            && let Some(Node::Expr(hir::Expr { hir_id, kind, .. })) =
                self.infcx.tcx.hir_get_if_local(def_id)
            && let ExprKind::Closure(hir::Closure { kind: hir::ClosureKind::Closure, .. }) = kind
            && let Node::Expr(expr) = self.infcx.tcx.parent_hir_node(*hir_id)
        {
            let mut cur_expr = expr;
            while let ExprKind::MethodCall(path_segment, recv, _, _) = cur_expr.kind {
                if path_segment.ident.name == sym::iter {
                    // Check that the type has an `iter_mut` method.
                    let res = self
                        .infcx
                        .tcx
                        .typeck(path_segment.hir_id.owner.def_id)
                        .type_dependent_def_id(cur_expr.hir_id)
                        .and_then(|def_id| self.infcx.tcx.impl_of_assoc(def_id))
                        .map(|def_id| self.infcx.tcx.associated_items(def_id))
                        .map(|assoc_items| {
                            assoc_items.filter_by_name_unhygienic(sym::iter_mut).peekable()
                        });

                    if let Some(mut res) = res
                        && res.peek().is_some()
                    {
                        err.span_suggestion_verbose(
                            path_segment.ident.span,
                            "you may want to use `iter_mut` here",
                            "iter_mut",
                            Applicability::MaybeIncorrect,
                        );
                    }
                    break;
                } else {
                    cur_expr = recv;
                }
            }
        }
    }

    fn suggest_make_local_mut(&self, err: &mut Diag<'_>, local: Local, name: Symbol) {
        let local_decl = &self.body.local_decls[local];

        let (pointer_sigil, pointer_desc) =
            if local_decl.ty.is_ref() { ("&", "reference") } else { ("*const", "pointer") };

        let (is_trait_sig, is_local, local_trait) = self.is_error_in_trait(local);

        if is_trait_sig && !is_local {
            // Do not suggest changing the signature when the trait comes from another crate.
            err.span_label(
                local_decl.source_info.span,
                format!("this is an immutable {pointer_desc}"),
            );
            return;
        }
        let decl_span = local_decl.source_info.span;

        let (amp_mut_sugg, local_var_ty_info) = match *local_decl.local_info() {
            LocalInfo::User(mir::BindingForm::ImplicitSelf(_)) => {
                let (span, suggestion) = suggest_ampmut_self(self.infcx.tcx, decl_span);
                let additional = local_trait.map(|span| suggest_ampmut_self(self.infcx.tcx, span));
                (AmpMutSugg::Type { span, suggestion, additional }, None)
            }

            LocalInfo::User(mir::BindingForm::Var(mir::VarBindingForm {
                binding_mode: BindingMode(ByRef::No, _),
                opt_ty_info,
                ..
            })) => {
                // Check if the RHS is from desugaring.
                let first_assignment = find_assignments(&self.body, local).first().copied();
                let first_assignment_stmt = first_assignment
                    .and_then(|loc| self.body[loc.block].statements.get(loc.statement_index));
                trace!(?first_assignment_stmt);
                let opt_assignment_rhs_span =
                    first_assignment.map(|loc| self.body.source_info(loc).span);
                let mut source_span = opt_assignment_rhs_span;
                if let Some(mir::Statement {
                    source_info: _,
                    kind:
                        mir::StatementKind::Assign(box (_, mir::Rvalue::Use(mir::Operand::Copy(place)))),
                    ..
                }) = first_assignment_stmt
                {
                    let local_span = self.body.local_decls[place.local].source_info.span;
                    // `&self` in async functions have a `desugaring_kind`, but the local we assign
                    // it with does not, so use the local_span for our checks later.
                    source_span = Some(local_span);
                    if let Some(DesugaringKind::ForLoop) = local_span.desugaring_kind() {
                        // On for loops, RHS points to the iterator part.
                        self.suggest_similar_mut_method_for_for_loop(err, local_span);
                        err.span_label(
                            local_span,
                            format!("this iterator yields `{pointer_sigil}` {pointer_desc}s",),
                        );
                        return;
                    }
                }

                // Don't create labels for compiler-generated spans or spans not from users' code.
                if source_span.is_some_and(|s| {
                    s.desugaring_kind().is_some() || self.infcx.tcx.sess.source_map().is_imported(s)
                }) {
                    return;
                }

                // This could be because we're in an `async fn`.
                if name == kw::SelfLower && opt_ty_info.is_none() {
                    let (span, suggestion) = suggest_ampmut_self(self.infcx.tcx, decl_span);
                    (AmpMutSugg::Type { span, suggestion, additional: None }, None)
                } else if let Some(sugg) =
                    suggest_ampmut(self.infcx, self.body(), first_assignment_stmt)
                {
                    (sugg, opt_ty_info)
                } else {
                    return;
                }
            }

            LocalInfo::User(mir::BindingForm::Var(mir::VarBindingForm {
                binding_mode: BindingMode(ByRef::Yes(_), _),
                ..
            })) => {
                let pattern_span: Span = local_decl.source_info.span;
                let Some(span) = suggest_ref_mut(self.infcx.tcx, pattern_span) else {
                    return;
                };
                (AmpMutSugg::Type { span, suggestion: "mut ".to_owned(), additional: None }, None)
            }

            _ => unreachable!(),
        };

        let mut suggest = |suggs: Vec<_>, applicability, extra| {
            if suggs.iter().any(|(span, _)| self.infcx.tcx.sess.source_map().is_imported(*span)) {
                return;
            }

            err.multipart_suggestion_verbose(
                format!(
                    "consider changing this to be a mutable {pointer_desc}{}{extra}",
                    if is_trait_sig {
                        " in the `impl` method and the `trait` definition"
                    } else {
                        ""
                    }
                ),
                suggs,
                applicability,
            );
        };

        let (mut sugg, add_type_annotation_if_not_exists) = match amp_mut_sugg {
            AmpMutSugg::Type { span, suggestion, additional } => {
                let mut sugg = vec![(span, suggestion)];
                sugg.extend(additional);
                suggest(sugg, Applicability::MachineApplicable, "");
                return;
            }
            AmpMutSugg::MapGetMut { span, suggestion } => {
                if self.infcx.tcx.sess.source_map().is_imported(span) {
                    return;
                }
                err.multipart_suggestion_verbose(
                    "consider using `get_mut`",
                    vec![(span, suggestion)],
                    Applicability::MaybeIncorrect,
                );
                return;
            }
            AmpMutSugg::Expr { span, suggestion } => {
                // `Expr` suggestions should change type annotations if they already exist (probably immut),
                // but do not add new type annotations.
                (vec![(span, suggestion)], false)
            }
            AmpMutSugg::ChangeBinding => (vec![], true),
        };

        // Find a binding's type to make mutable.
        let (binding_exists, span) = match local_var_ty_info {
            // If this is a variable binding with an explicit type,
            // then we will suggest changing it to be mutable.
            // This is `Applicability::MachineApplicable`.
            Some(ty_span) => (true, ty_span),

            // Otherwise, we'll suggest *adding* an annotated type, we'll suggest
            // the RHS's type for that.
            // This is `Applicability::HasPlaceholders`.
            None => (false, decl_span),
        };

        if !binding_exists && !add_type_annotation_if_not_exists {
            suggest(sugg, Applicability::MachineApplicable, "");
            return;
        }

        // If the binding already exists and is a reference with an explicit
        // lifetime, then we can suggest adding ` mut`. This is special-cased from
        // the path without an explicit lifetime.
        let (sugg_span, sugg_str, suggest_now) = if let Ok(src) = self.infcx.tcx.sess.source_map().span_to_snippet(span)
            && src.starts_with("&'")
            // Note that `&' a T` is invalid so this is correct.
            && let Some(ws_pos) = src.find(char::is_whitespace)
        {
            let span = span.with_lo(span.lo() + BytePos(ws_pos as u32)).shrink_to_lo();
            (span, " mut".to_owned(), true)
        // If there is already a binding, we modify it to be `mut`.
        } else if binding_exists {
            // Shrink the span to just after the `&` in `&variable`.
            let span = span.with_lo(span.lo() + BytePos(1)).shrink_to_lo();
            (span, "mut ".to_owned(), true)
        } else {
            // Otherwise, suggest that the user annotates the binding; We provide the
            // type of the local.
            let ty = local_decl.ty.builtin_deref(true).unwrap();

            (span, format!("{}mut {}", if local_decl.ty.is_ref() { "&" } else { "*" }, ty), false)
        };

        if suggest_now {
            // Suggest changing `&x` to `&mut x` and changing `&T` to `&mut T` at the same time.
            let has_change = !sugg.is_empty();
            sugg.push((sugg_span, sugg_str));
            suggest(
                sugg,
                Applicability::MachineApplicable,
                // FIXME(fee1-dead) this somehow doesn't fire
                if has_change { " and changing the binding's type" } else { "" },
            );
            return;
        } else if !sugg.is_empty() {
            suggest(sugg, Applicability::MachineApplicable, "");
            return;
        }

        let def_id = self.body.source.def_id();
        let hir_id = if let Some(local_def_id) = def_id.as_local()
            && let Some(body) = self.infcx.tcx.hir_maybe_body_owned_by(local_def_id)
        {
            BindingFinder { span: sugg_span }.visit_body(&body).break_value()
        } else {
            None
        };
        let node = hir_id.map(|hir_id| self.infcx.tcx.hir_node(hir_id));

        let Some(hir::Node::LetStmt(local)) = node else {
            err.span_label(
                sugg_span,
                format!("consider changing this binding's type to be: `{sugg_str}`"),
            );
            return;
        };

        let tables = self.infcx.tcx.typeck(def_id.as_local().unwrap());
        if let Some(clone_trait) = self.infcx.tcx.lang_items().clone_trait()
            && let Some(expr) = local.init
            && let ty = tables.node_type_opt(expr.hir_id)
            && let Some(ty) = ty
            && let ty::Ref(..) = ty.kind()
        {
            match self
                .infcx
                .type_implements_trait_shallow(clone_trait, ty.peel_refs(), self.infcx.param_env)
                .as_deref()
            {
                Some([]) => {
                    // FIXME: This error message isn't useful, since we're just
                    // vaguely suggesting to clone a value that already
                    // implements `Clone`.
                    //
                    // A correct suggestion here would take into account the fact
                    // that inference may be affected by missing types on bindings,
                    // etc., to improve "tests/ui/borrowck/issue-91206.stderr", for
                    // example.
                }
                None => {
                    if let hir::ExprKind::MethodCall(segment, _rcvr, [], span) = expr.kind
                        && segment.ident.name == sym::clone
                    {
                        err.span_help(
                            span,
                            format!(
                                "`{}` doesn't implement `Clone`, so this call clones \
                                             the reference `{ty}`",
                                ty.peel_refs(),
                            ),
                        );
                    }
                    // The type doesn't implement Clone.
                    let trait_ref = ty::Binder::dummy(ty::TraitRef::new(
                        self.infcx.tcx,
                        clone_trait,
                        [ty.peel_refs()],
                    ));
                    let obligation = traits::Obligation::new(
                        self.infcx.tcx,
                        traits::ObligationCause::dummy(),
                        self.infcx.param_env,
                        trait_ref,
                    );
                    self.infcx.err_ctxt().suggest_derive(
                        &obligation,
                        err,
                        trait_ref.upcast(self.infcx.tcx),
                    );
                }
                Some(errors) => {
                    if let hir::ExprKind::MethodCall(segment, _rcvr, [], span) = expr.kind
                        && segment.ident.name == sym::clone
                    {
                        err.span_help(
                            span,
                            format!(
                                "`{}` doesn't implement `Clone` because its \
                                             implementations trait bounds could not be met, so \
                                             this call clones the reference `{ty}`",
                                ty.peel_refs(),
                            ),
                        );
                        err.note(format!(
                            "the following trait bounds weren't met: {}",
                            errors
                                .iter()
                                .map(|e| e.obligation.predicate.to_string())
                                .collect::<Vec<_>>()
                                .join("\n"),
                        ));
                    }
                    // The type doesn't implement Clone because of unmet obligations.
                    for error in errors {
                        if let traits::FulfillmentErrorCode::Select(
                            traits::SelectionError::Unimplemented,
                        ) = error.code
                            && let ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred)) =
                                error.obligation.predicate.kind().skip_binder()
                        {
                            self.infcx.err_ctxt().suggest_derive(
                                &error.obligation,
                                err,
                                error.obligation.predicate.kind().rebind(pred),
                            );
                        }
                    }
                }
            }
        }
        let (changing, span, sugg) = match local.ty {
            Some(ty) => ("changing", ty.span, sugg_str),
            None => ("specifying", local.pat.span.shrink_to_hi(), format!(": {sugg_str}")),
        };
        err.span_suggestion_verbose(
            span,
            format!("consider {changing} this binding's type"),
            sugg,
            Applicability::HasPlaceholders,
        );
    }
}

struct BindingFinder {
    span: Span,
}

impl<'tcx> Visitor<'tcx> for BindingFinder {
    type Result = ControlFlow<hir::HirId>;
    fn visit_stmt(&mut self, s: &'tcx hir::Stmt<'tcx>) -> Self::Result {
        if let hir::StmtKind::Let(local) = s.kind
            && local.pat.span == self.span
        {
            ControlFlow::Break(local.hir_id)
        } else {
            hir::intravisit::walk_stmt(self, s)
        }
    }

    fn visit_param(&mut self, param: &'tcx hir::Param<'tcx>) -> Self::Result {
        if let hir::Pat { kind: hir::PatKind::Ref(_, _), span, .. } = param.pat
            && *span == self.span
        {
            ControlFlow::Break(param.hir_id)
        } else {
            ControlFlow::Continue(())
        }
    }
}

fn mut_borrow_of_mutable_ref(local_decl: &LocalDecl<'_>, local_name: Option<Symbol>) -> bool {
    debug!("local_info: {:?}, ty.kind(): {:?}", local_decl.local_info, local_decl.ty.kind());

    match *local_decl.local_info() {
        // Check if mutably borrowing a mutable reference.
        LocalInfo::User(mir::BindingForm::Var(mir::VarBindingForm {
            binding_mode: BindingMode(ByRef::No, Mutability::Not),
            ..
        })) => matches!(local_decl.ty.kind(), ty::Ref(_, _, hir::Mutability::Mut)),
        LocalInfo::User(mir::BindingForm::ImplicitSelf(kind)) => {
            // Check if the user variable is a `&mut self` and we can therefore
            // suggest removing the `&mut`.
            //
            // Deliberately fall into this case for all implicit self types,
            // so that we don't fall into the next case with them.
            kind == hir::ImplicitSelfKind::RefMut
        }
        _ if Some(kw::SelfLower) == local_name => {
            // Otherwise, check if the name is the `self` keyword - in which case
            // we have an explicit self. Do the same thing in this case and check
            // for a `self: &mut Self` to suggest removing the `&mut`.
            matches!(local_decl.ty.kind(), ty::Ref(_, _, hir::Mutability::Mut))
        }
        _ => false,
    }
}

fn suggest_ampmut_self(tcx: TyCtxt<'_>, span: Span) -> (Span, String) {
    match tcx.sess.source_map().span_to_snippet(span) {
        Ok(snippet) if snippet.ends_with("self") => {
            (span.with_hi(span.hi() - BytePos(4)).shrink_to_hi(), "mut ".to_string())
        }
        _ => (span, "&mut self".to_string()),
    }
}

enum AmpMutSugg {
    /// Type suggestion. Changes `&self` to `&mut self`, `x: &T` to `x: &mut T`,
    /// `ref x` to `ref mut x`, etc.
    Type {
        span: Span,
        suggestion: String,
        additional: Option<(Span, String)>,
    },
    /// Suggestion for expressions, `&x` to `&mut x`, `&x[i]` to `&mut x[i]`, etc.
    Expr {
        span: Span,
        suggestion: String,
    },
    /// Suggests `.get_mut` in the case of `&map[&key]` for Hash/BTreeMap.
    MapGetMut {
        span: Span,
        suggestion: String,
    },
    ChangeBinding,
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
    infcx: &crate::BorrowckInferCtxt<'tcx>,
    body: &Body<'tcx>,
    opt_assignment_rhs_stmt: Option<&Statement<'tcx>>,
) -> Option<AmpMutSugg> {
    let tcx = infcx.tcx;
    // If there is a RHS and it starts with a `&` from it, then check if it is
    // mutable, and if not, put suggest putting `mut ` to make it mutable.
    // We don't have to worry about lifetime annotations here because they are
    // not valid when taking a reference. For example, the following is not valid Rust:
    //
    // let x: &i32 = &'a 5;
    //                ^^ lifetime annotation not allowed
    //
    if let Some(rhs_stmt) = opt_assignment_rhs_stmt
        && let StatementKind::Assign(box (lhs, rvalue)) = &rhs_stmt.kind
        && let mut rhs_span = rhs_stmt.source_info.span
        && let Ok(mut rhs_str) = tcx.sess.source_map().span_to_snippet(rhs_span)
    {
        let mut rvalue = rvalue;

        // Take some special care when handling `let _x = &*_y`:
        // We want to know if this is part of an overloaded index, so `let x = &a[0]`,
        // or whether this is a usertype ascription (`let _x: &T = y`).
        if let Rvalue::Ref(_, BorrowKind::Shared, place) = rvalue
            && place.projection.len() == 1
            && place.projection[0] == ProjectionElem::Deref
            && let Some(assign) = find_assignments(&body, place.local).first()
        {
            // If this is a usertype ascription (`let _x: &T = _y`) then pierce through it as either we want
            // to suggest `&mut` on the expression (handled here) or we return `None` and let the caller
            // suggest `&mut` on the type if the expression seems fine (e.g. `let _x: &T = &mut _y`).
            if let Some(user_ty_projs) = body.local_decls[lhs.local].user_ty.as_ref()
                && let [user_ty_proj] = user_ty_projs.contents.as_slice()
                && user_ty_proj.projs.is_empty()
                && let Either::Left(rhs_stmt_new) = body.stmt_at(*assign)
                && let StatementKind::Assign(box (_, rvalue_new)) = &rhs_stmt_new.kind
                && let rhs_span_new = rhs_stmt_new.source_info.span
                && let Ok(rhs_str_new) = tcx.sess.source_map().span_to_snippet(rhs_span)
            {
                (rvalue, rhs_span, rhs_str) = (rvalue_new, rhs_span_new, rhs_str_new);
            }

            if let Either::Right(call) = body.stmt_at(*assign)
                && let TerminatorKind::Call {
                    func: Operand::Constant(box const_operand), args, ..
                } = &call.kind
                && let ty::FnDef(method_def_id, method_args) = *const_operand.ty().kind()
                && let Some(trait_) = tcx.trait_of_assoc(method_def_id)
                && tcx.is_lang_item(trait_, hir::LangItem::Index)
            {
                let trait_ref = ty::TraitRef::from_assoc(
                    tcx,
                    tcx.require_lang_item(hir::LangItem::IndexMut, rhs_span),
                    method_args,
                );
                // The type only implements `Index` but not `IndexMut`, we must not suggest `&mut`.
                if !infcx
                    .type_implements_trait(trait_ref.def_id, trait_ref.args, infcx.param_env)
                    .must_apply_considering_regions()
                {
                    // Suggest `get_mut` if type is a `BTreeMap` or `HashMap`.
                    if let ty::Adt(def, _) = trait_ref.self_ty().kind()
                        && [sym::BTreeMap, sym::HashMap]
                            .into_iter()
                            .any(|s| tcx.is_diagnostic_item(s, def.did()))
                        && let [map, key] = &**args
                        && let Ok(map) = tcx.sess.source_map().span_to_snippet(map.span)
                        && let Ok(key) = tcx.sess.source_map().span_to_snippet(key.span)
                    {
                        let span = rhs_span;
                        let suggestion = format!("{map}.get_mut({key}).unwrap()");
                        return Some(AmpMutSugg::MapGetMut { span, suggestion });
                    }
                    return None;
                }
            }
        }

        let sugg = match rvalue {
            Rvalue::Ref(_, BorrowKind::Shared, _) if let Some(ref_idx) = rhs_str.find('&') => {
                // Shrink the span to just after the `&` in `&variable`.
                Some((
                    rhs_span.with_lo(rhs_span.lo() + BytePos(ref_idx as u32 + 1)).shrink_to_lo(),
                    "mut ".to_owned(),
                ))
            }
            Rvalue::RawPtr(RawPtrKind::Const, _) if let Some(const_idx) = rhs_str.find("const") => {
                // Suggest changing `&raw const` to `&raw mut` if applicable.
                let const_idx = const_idx as u32;
                Some((
                    rhs_span
                        .with_lo(rhs_span.lo() + BytePos(const_idx))
                        .with_hi(rhs_span.lo() + BytePos(const_idx + "const".len() as u32)),
                    "mut".to_owned(),
                ))
            }
            _ => None,
        };

        if let Some((span, suggestion)) = sugg {
            return Some(AmpMutSugg::Expr { span, suggestion });
        }
    }

    Some(AmpMutSugg::ChangeBinding)
}

/// If the type is a `Coroutine`, `Closure`, or `CoroutineClosure`
fn is_closure_like(ty: Ty<'_>) -> bool {
    ty.is_closure() || ty.is_coroutine() || ty.is_coroutine_closure()
}

/// Given a field that needs to be mutable, returns a span where the " mut " could go.
/// This function expects the local to be a reference to a struct in order to produce a span.
///
/// ```text
/// LL |     s: &'a   String
///    |           ^^^ returns a span taking up the space here
/// ```
fn get_mut_span_in_struct_field<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    field: FieldIdx,
) -> Option<Span> {
    // Expect our local to be a reference to a struct of some kind.
    if let ty::Ref(_, ty, _) = ty.kind()
        && let ty::Adt(def, _) = ty.kind()
        && let field = def.all_fields().nth(field.index())?
        // Now we're dealing with the actual struct that we're going to suggest a change to,
        // we can expect a field that is an immutable reference to a type.
        && let hir::Node::Field(field) = tcx.hir_node_by_def_id(field.did.as_local()?)
        && let hir::TyKind::Ref(lt, hir::MutTy { mutbl: hir::Mutability::Not, ty }) = field.ty.kind
    {
        return Some(lt.ident.span.between(ty.span));
    }

    None
}

/// If possible, suggest replacing `ref` with `ref mut`.
fn suggest_ref_mut(tcx: TyCtxt<'_>, span: Span) -> Option<Span> {
    let pattern_str = tcx.sess.source_map().span_to_snippet(span).ok()?;
    if let Some(rest) = pattern_str.strip_prefix("ref")
        && rest.starts_with(rustc_lexer::is_whitespace)
    {
        let span = span.with_lo(span.lo() + BytePos(4)).shrink_to_lo();
        Some(span)
    } else {
        None
    }
}
