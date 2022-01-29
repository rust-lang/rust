use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_hir_and_then};
use clippy_utils::source::{snippet_with_applicability, snippet_with_context};
use clippy_utils::sugg::has_enclosing_paren;
use clippy_utils::ty::{expr_sig, peel_mid_ty_refs, variant_of_res};
use clippy_utils::{get_parent_expr, is_lint_allowed, path_to_local, walk_to_expr_usage};
use rustc_ast::util::parser::{PREC_POSTFIX, PREC_PREFIX};
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::Applicability;
use rustc_hir::{
    self as hir, BindingAnnotation, Body, BodyId, BorrowKind, Expr, ExprKind, FnRetTy, GenericArg, HirId, ImplItem,
    ImplItemKind, Item, ItemKind, Local, MatchSource, Mutability, Node, Pat, PatKind, Path, QPath, TraitItem,
    TraitItemKind, TyKind, UnOp,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable, TypeckResults};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{symbol::sym, Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for explicit `deref()` or `deref_mut()` method calls.
    ///
    /// ### Why is this bad?
    /// Dereferencing by `&*x` or `&mut *x` is clearer and more concise,
    /// when not part of a method chain.
    ///
    /// ### Example
    /// ```rust
    /// use std::ops::Deref;
    /// let a: &mut String = &mut String::from("foo");
    /// let b: &str = a.deref();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let a: &mut String = &mut String::from("foo");
    /// let b = &*a;
    /// ```
    ///
    /// This lint excludes:
    /// ```rust,ignore
    /// let _ = d.unwrap().deref();
    /// ```
    #[clippy::version = "1.44.0"]
    pub EXPLICIT_DEREF_METHODS,
    pedantic,
    "Explicit use of deref or deref_mut method while not in a method chain."
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for address of operations (`&`) that are going to
    /// be dereferenced immediately by the compiler.
    ///
    /// ### Why is this bad?
    /// Suggests that the receiver of the expression borrows
    /// the expression.
    ///
    /// ### Example
    /// ```rust
    /// fn fun(_a: &i32) {}
    ///
    /// let x: &i32 = &&&&&&5;
    /// fun(&x);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # fn fun(_a: &i32) {}
    /// let x: &i32 = &5;
    /// fun(x);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEEDLESS_BORROW,
    style,
    "taking a reference that is going to be automatically dereferenced"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `ref` bindings which create a reference to a reference.
    ///
    /// ### Why is this bad?
    /// The address-of operator at the use site is clearer about the need for a reference.
    ///
    /// ### Example
    /// ```rust
    /// let x = Some("");
    /// if let Some(ref x) = x {
    ///     // use `x` here
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let x = Some("");
    /// if let Some(x) = x {
    ///     // use `&x` here
    /// }
    /// ```
    #[clippy::version = "1.54.0"]
    pub REF_BINDING_TO_REFERENCE,
    pedantic,
    "`ref` binding to a reference"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for dereferencing expressions which would be covered by auto-deref.
    ///
    /// ### Why is this bad?
    /// This unnecessarily complicates the code.
    ///
    /// ### Example
    /// ```rust
    /// let x = String::new();
    /// let y: &str = &*x;
    /// ```
    /// Use instead:
    /// ```rust
    /// let x = String::new();
    /// let y: &str = &x;
    /// ```
    #[clippy::version = "1.60.0"]
    pub EXPLICIT_AUTO_DEREF,
    complexity,
    "dereferencing when the compiler would automatically dereference"
}

impl_lint_pass!(Dereferencing => [
    EXPLICIT_DEREF_METHODS,
    NEEDLESS_BORROW,
    REF_BINDING_TO_REFERENCE,
    EXPLICIT_AUTO_DEREF,
]);

#[derive(Default)]
pub struct Dereferencing {
    state: Option<(State, StateData)>,

    // While parsing a `deref` method call in ufcs form, the path to the function is itself an
    // expression. This is to store the id of that expression so it can be skipped when
    // `check_expr` is called for it.
    skip_expr: Option<HirId>,

    /// The body the first local was found in. Used to emit lints when the traversal of the body has
    /// been finished. Note we can't lint at the end of every body as they can be nested within each
    /// other.
    current_body: Option<BodyId>,
    /// The list of locals currently being checked by the lint.
    /// If the value is `None`, then the binding has been seen as a ref pattern, but is not linted.
    /// This is needed for or patterns where one of the branches can be linted, but another can not
    /// be.
    ///
    /// e.g. `m!(x) | Foo::Bar(ref x)`
    ref_locals: FxIndexMap<HirId, Option<RefPat>>,
}

struct StateData {
    /// Span of the top level expression
    span: Span,
    hir_id: HirId,
}

struct DerefedBorrow {
    count: usize,
    required_precedence: i8,
    msg: &'static str,
    position: Position,
}

enum State {
    // Any number of deref method calls.
    DerefMethod {
        // The number of calls in a sequence which changed the referenced type
        ty_changed_count: usize,
        is_final_ufcs: bool,
        /// The required mutability
        target_mut: Mutability,
    },
    DerefedBorrow(DerefedBorrow),
    ExplicitDeref {
        deref_span: Span,
        deref_hir_id: HirId,
    },
    ExplicitDerefField {
        name: Symbol,
    },
    Reborrow {
        deref_span: Span,
        deref_hir_id: HirId,
    },
    Borrow,
}

// A reference operation considered by this lint pass
enum RefOp {
    Method(Mutability),
    Deref,
    AddrOf,
}

struct RefPat {
    /// Whether every usage of the binding is dereferenced.
    always_deref: bool,
    /// The spans of all the ref bindings for this local.
    spans: Vec<Span>,
    /// The applicability of this suggestion.
    app: Applicability,
    /// All the replacements which need to be made.
    replacements: Vec<(Span, String)>,
    /// The [`HirId`] that the lint should be emitted at.
    hir_id: HirId,
}

impl<'tcx> LateLintPass<'tcx> for Dereferencing {
    #[expect(clippy::too_many_lines)]
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // Skip path expressions from deref calls. e.g. `Deref::deref(e)`
        if Some(expr.hir_id) == self.skip_expr.take() {
            return;
        }

        if let Some(local) = path_to_local(expr) {
            self.check_local_usage(cx, expr, local);
        }

        // Stop processing sub expressions when a macro call is seen
        if expr.span.from_expansion() {
            if let Some((state, data)) = self.state.take() {
                report(cx, expr, state, data);
            }
            return;
        }

        let typeck = cx.typeck_results();
        let (kind, sub_expr) = if let Some(x) = try_parse_ref_op(cx.tcx, typeck, expr) {
            x
        } else {
            // The whole chain of reference operations has been seen
            if let Some((state, data)) = self.state.take() {
                report(cx, expr, state, data);
            }
            return;
        };

        match (self.state.take(), kind) {
            (None, kind) => {
                let expr_ty = typeck.expr_ty(expr);
                let (position, adjustments) = walk_parents(cx, expr);

                match kind {
                    RefOp::Deref => {
                        if let Position::FieldAccess(name) = position
                            && !ty_contains_field(typeck.expr_ty(sub_expr), name)
                        {
                            self.state = Some((
                                State::ExplicitDerefField { name },
                                StateData { span: expr.span, hir_id: expr.hir_id },
                            ));
                        } else if position.is_deref_stable() {
                            self.state = Some((
                                State::ExplicitDeref { deref_span: expr.span, deref_hir_id: expr.hir_id },
                                StateData { span: expr.span, hir_id: expr.hir_id },
                            ));
                        }
                    }
                    RefOp::Method(target_mut)
                        if !is_lint_allowed(cx, EXPLICIT_DEREF_METHODS, expr.hir_id)
                            && position.lint_explicit_deref() =>
                    {
                        self.state = Some((
                            State::DerefMethod {
                                ty_changed_count: if deref_method_same_type(expr_ty, typeck.expr_ty(sub_expr)) {
                                    0
                                } else {
                                    1
                                },
                                is_final_ufcs: matches!(expr.kind, ExprKind::Call(..)),
                                target_mut,
                            },
                            StateData {
                                span: expr.span,
                                hir_id: expr.hir_id,
                            },
                        ));
                    },
                    RefOp::AddrOf => {
                        // Find the number of times the borrow is auto-derefed.
                        let mut iter = adjustments.iter();
                        let mut deref_count = 0usize;
                        let next_adjust = loop {
                            match iter.next() {
                                Some(adjust) => {
                                    if !matches!(adjust.kind, Adjust::Deref(_)) {
                                        break Some(adjust);
                                    } else if !adjust.target.is_ref() {
                                        deref_count += 1;
                                        break iter.next();
                                    }
                                    deref_count += 1;
                                },
                                None => break None,
                            };
                        };

                        // Determine the required number of references before any can be removed. In all cases the
                        // reference made by the current expression will be removed. After that there are four cases to
                        // handle.
                        //
                        // 1. Auto-borrow will trigger in the current position, so no further references are required.
                        // 2. Auto-deref ends at a reference, or the underlying type, so one extra needs to be left to
                        //    handle the automatically inserted re-borrow.
                        // 3. Auto-deref hits a user-defined `Deref` impl, so at least one reference needs to exist to
                        //    start auto-deref.
                        // 4. If the chain of non-user-defined derefs ends with a mutable re-borrow, and re-borrow
                        //    adjustments will not be inserted automatically, then leave one further reference to avoid
                        //    moving a mutable borrow.
                        //    e.g.
                        //        fn foo<T>(x: &mut Option<&mut T>, y: &mut T) {
                        //            let x = match x {
                        //                // Removing the borrow will cause `x` to be moved
                        //                Some(x) => &mut *x,
                        //                None => y
                        //            };
                        //        }
                        let deref_msg =
                            "this expression creates a reference which is immediately dereferenced by the compiler";
                        let borrow_msg = "this expression borrows a value the compiler would automatically borrow";

                        let (required_refs, required_precedence, msg) = if position.can_auto_borrow() {
                            (1, PREC_POSTFIX, if deref_count == 1 { borrow_msg } else { deref_msg })
                        } else if let Some(&Adjust::Borrow(AutoBorrow::Ref(_, mutability))) =
                            next_adjust.map(|a| &a.kind)
                        {
                            if matches!(mutability, AutoBorrowMutability::Mut { .. }) && !position.is_reborrow_stable()
                            {
                                (3, 0, deref_msg)
                            } else {
                                (2, 0, deref_msg)
                            }
                        } else {
                            (2, 0, deref_msg)
                        };

                        if deref_count >= required_refs {
                            self.state = Some((
                                State::DerefedBorrow(DerefedBorrow {
                                    // One of the required refs is for the current borrow expression, the remaining ones
                                    // can't be removed without breaking the code. See earlier comment.
                                    count: deref_count - required_refs,
                                    required_precedence,
                                    msg,
                                    position,
                                }),
                                StateData { span: expr.span, hir_id: expr.hir_id },
                            ));
                        } else if position.is_deref_stable() {
                            self.state = Some((
                                State::Borrow,
                                StateData {
                                    span: expr.span,
                                    hir_id: expr.hir_id,
                                },
                            ));
                        }
                    },
                    RefOp::Method(..) => (),
                }
            },
            (
                Some((
                    State::DerefMethod {
                        target_mut,
                        ty_changed_count,
                        ..
                    },
                    data,
                )),
                RefOp::Method(_),
            ) => {
                self.state = Some((
                    State::DerefMethod {
                        ty_changed_count: if deref_method_same_type(typeck.expr_ty(expr), typeck.expr_ty(sub_expr)) {
                            ty_changed_count
                        } else {
                            ty_changed_count + 1
                        },
                        is_final_ufcs: matches!(expr.kind, ExprKind::Call(..)),
                        target_mut,
                    },
                    data,
                ));
            },
            (Some((State::DerefedBorrow(state), data)), RefOp::AddrOf) if state.count != 0 => {
                self.state = Some((
                    State::DerefedBorrow(DerefedBorrow {
                        count: state.count - 1,
                        ..state
                    }),
                    data,
                ));
            },
            (Some((State::DerefedBorrow(state), data)), RefOp::AddrOf) => {
                let position = state.position;
                report(cx, expr, State::DerefedBorrow(state), data);
                if position.is_deref_stable() {
                    self.state = Some((
                        State::Borrow,
                        StateData {
                            span: expr.span,
                            hir_id: expr.hir_id,
                        },
                    ));
                }
            },
            (Some((State::DerefedBorrow(state), data)), RefOp::Deref) => {
                let position = state.position;
                report(cx, expr, State::DerefedBorrow(state), data);
                if let Position::FieldAccess(name) = position
                    && !ty_contains_field(typeck.expr_ty(sub_expr), name)
                {
                    self.state = Some((
                        State::ExplicitDerefField { name },
                        StateData { span: expr.span, hir_id: expr.hir_id },
                    ));
                } else if position.is_deref_stable() {
                    self.state = Some((
                        State::ExplicitDeref { deref_span: expr.span, deref_hir_id: expr.hir_id },
                        StateData { span: expr.span, hir_id: expr.hir_id },
                    ));
                }
            },

            (Some((State::Borrow, data)), RefOp::Deref) => {
                if typeck.expr_ty(sub_expr).is_ref() {
                    self.state = Some((
                        State::Reborrow {
                            deref_span: expr.span,
                            deref_hir_id: expr.hir_id,
                        },
                        data,
                    ));
                } else {
                    self.state = Some((
                        State::ExplicitDeref {
                            deref_span: expr.span,
                            deref_hir_id: expr.hir_id,
                        },
                        data,
                    ));
                }
            },
            (
                Some((
                    State::Reborrow {
                        deref_span,
                        deref_hir_id,
                    },
                    data,
                )),
                RefOp::Deref,
            ) => {
                self.state = Some((
                    State::ExplicitDeref {
                        deref_span,
                        deref_hir_id,
                    },
                    data,
                ));
            },
            (state @ Some((State::ExplicitDeref { .. }, _)), RefOp::Deref) => {
                self.state = state;
            },
            (Some((State::ExplicitDerefField { name }, data)), RefOp::Deref)
                if !ty_contains_field(typeck.expr_ty(sub_expr), name) =>
            {
                self.state = Some((State::ExplicitDerefField { name }, data));
            },

            (Some((state, data)), _) => report(cx, expr, state, data),
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>) {
        if let PatKind::Binding(BindingAnnotation::Ref, id, name, _) = pat.kind {
            if let Some(opt_prev_pat) = self.ref_locals.get_mut(&id) {
                // This binding id has been seen before. Add this pattern to the list of changes.
                if let Some(prev_pat) = opt_prev_pat {
                    if pat.span.from_expansion() {
                        // Doesn't match the context of the previous pattern. Can't lint here.
                        *opt_prev_pat = None;
                    } else {
                        prev_pat.spans.push(pat.span);
                        prev_pat.replacements.push((
                            pat.span,
                            snippet_with_context(cx, name.span, pat.span.ctxt(), "..", &mut prev_pat.app)
                                .0
                                .into(),
                        ));
                    }
                }
                return;
            }

            if_chain! {
                if !pat.span.from_expansion();
                if let ty::Ref(_, tam, _) = *cx.typeck_results().pat_ty(pat).kind();
                // only lint immutable refs, because borrowed `&mut T` cannot be moved out
                if let ty::Ref(_, _, Mutability::Not) = *tam.kind();
                then {
                    let mut app = Applicability::MachineApplicable;
                    let snip = snippet_with_context(cx, name.span, pat.span.ctxt(), "..", &mut app).0;
                    self.current_body = self.current_body.or(cx.enclosing_body);
                    self.ref_locals.insert(
                        id,
                        Some(RefPat {
                            always_deref: true,
                            spans: vec![pat.span],
                            app,
                            replacements: vec![(pat.span, snip.into())],
                            hir_id: pat.hir_id
                        }),
                    );
                }
            }
        }
    }

    fn check_body_post(&mut self, cx: &LateContext<'tcx>, body: &'tcx Body<'_>) {
        if Some(body.id()) == self.current_body {
            for pat in self.ref_locals.drain(..).filter_map(|(_, x)| x) {
                let replacements = pat.replacements;
                let app = pat.app;
                let lint = if pat.always_deref {
                    NEEDLESS_BORROW
                } else {
                    REF_BINDING_TO_REFERENCE
                };
                span_lint_hir_and_then(
                    cx,
                    lint,
                    pat.hir_id,
                    pat.spans,
                    "this pattern creates a reference to a reference",
                    |diag| {
                        diag.multipart_suggestion("try this", replacements, app);
                    },
                );
            }
            self.current_body = None;
        }
    }
}

fn try_parse_ref_op<'tcx>(
    tcx: TyCtxt<'tcx>,
    typeck: &'tcx TypeckResults<'_>,
    expr: &'tcx Expr<'_>,
) -> Option<(RefOp, &'tcx Expr<'tcx>)> {
    let (def_id, arg) = match expr.kind {
        ExprKind::MethodCall(_, [arg], _) => (typeck.type_dependent_def_id(expr.hir_id)?, arg),
        ExprKind::Call(
            Expr {
                kind: ExprKind::Path(path),
                hir_id,
                ..
            },
            [arg],
        ) => (typeck.qpath_res(path, *hir_id).opt_def_id()?, arg),
        ExprKind::Unary(UnOp::Deref, sub_expr) if !typeck.expr_ty(sub_expr).is_unsafe_ptr() => {
            return Some((RefOp::Deref, sub_expr));
        },
        ExprKind::AddrOf(BorrowKind::Ref, _, sub_expr) => return Some((RefOp::AddrOf, sub_expr)),
        _ => return None,
    };
    if tcx.is_diagnostic_item(sym::deref_method, def_id) {
        Some((RefOp::Method(Mutability::Not), arg))
    } else if tcx.trait_of_item(def_id)? == tcx.lang_items().deref_mut_trait()? {
        Some((RefOp::Method(Mutability::Mut), arg))
    } else {
        None
    }
}

// Checks whether the type for a deref call actually changed the type, not just the mutability of
// the reference.
fn deref_method_same_type<'tcx>(result_ty: Ty<'tcx>, arg_ty: Ty<'tcx>) -> bool {
    match (result_ty.kind(), arg_ty.kind()) {
        (ty::Ref(_, result_ty, _), ty::Ref(_, arg_ty, _)) => result_ty == arg_ty,

        // The result type for a deref method is always a reference
        // Not matching the previous pattern means the argument type is not a reference
        // This means that the type did change
        _ => false,
    }
}

/// The position of an expression relative to it's parent.
#[derive(Clone, Copy)]
enum Position {
    MethodReceiver,
    Callee,
    FieldAccess(Symbol),
    Postfix,
    Deref,
    /// Any other location which will trigger auto-deref to a specific time.
    DerefStable,
    /// Any other location which will trigger auto-reborrowing.
    ReborrowStable,
    Other,
}
impl Position {
    fn is_deref_stable(self) -> bool {
        matches!(self, Self::DerefStable)
    }

    fn is_reborrow_stable(self) -> bool {
        matches!(self, Self::DerefStable | Self::ReborrowStable)
    }

    fn can_auto_borrow(self) -> bool {
        matches!(self, Self::MethodReceiver | Self::FieldAccess(_) | Self::Callee)
    }

    fn lint_explicit_deref(self) -> bool {
        matches!(self, Self::Other | Self::DerefStable | Self::ReborrowStable)
    }
}

/// Walks up the parent expressions attempting to determine both how stable the auto-deref result
/// is, and which adjustments will be applied to it. Note this will not consider auto-borrow
/// locations as those follow different rules.
#[allow(clippy::too_many_lines)]
fn walk_parents<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) -> (Position, &'tcx [Adjustment<'tcx>]) {
    let mut adjustments = [].as_slice();
    let ctxt = e.span.ctxt();
    let position = walk_to_expr_usage(cx, e, &mut |parent, child_id| {
        // LocalTableInContext returns the wrong lifetime, so go use `expr_adjustments` instead.
        if adjustments.is_empty() && let Node::Expr(e) = cx.tcx.hir().get(child_id) {
            adjustments = cx.typeck_results().expr_adjustments(e);
        }
        match parent {
            Node::Local(Local { ty: Some(ty), span, .. }) if span.ctxt() == ctxt => {
                Some(binding_ty_auto_deref_stability(ty))
            },
            Node::Item(&Item {
                kind: ItemKind::Static(..) | ItemKind::Const(..),
                def_id,
                span,
                ..
            })
            | Node::TraitItem(&TraitItem {
                kind: TraitItemKind::Const(..),
                def_id,
                span,
                ..
            })
            | Node::ImplItem(&ImplItem {
                kind: ImplItemKind::Const(..),
                def_id,
                span,
                ..
            }) if span.ctxt() == ctxt => {
                let ty = cx.tcx.type_of(def_id);
                Some(if ty.is_ref() {
                    Position::Other
                } else {
                    Position::DerefStable
                })
            },

            Node::Item(&Item {
                kind: ItemKind::Fn(..),
                def_id,
                span,
                ..
            })
            | Node::TraitItem(&TraitItem {
                kind: TraitItemKind::Fn(..),
                def_id,
                span,
                ..
            })
            | Node::ImplItem(&ImplItem {
                kind: ImplItemKind::Fn(..),
                def_id,
                span,
                ..
            }) if span.ctxt() == ctxt => {
                let output = cx.tcx.fn_sig(def_id.to_def_id()).skip_binder().output();
                Some(if !output.is_ref() {
                    Position::Other
                } else if output.has_placeholders() || output.has_opaque_types() {
                    Position::ReborrowStable
                } else {
                    Position::DerefStable
                })
            },

            Node::Expr(parent) if parent.span.ctxt() == ctxt => match parent.kind {
                ExprKind::Ret(_) => {
                    let output = cx
                        .tcx
                        .fn_sig(cx.tcx.hir().body_owner_def_id(cx.enclosing_body.unwrap()))
                        .skip_binder()
                        .output();
                    Some(if !output.is_ref() {
                        Position::Other
                    } else if output.has_placeholders() || output.has_opaque_types() {
                        Position::ReborrowStable
                    } else {
                        Position::DerefStable
                    })
                },
                ExprKind::Call(func, _) if func.hir_id == child_id => (child_id == e.hir_id).then(|| Position::Callee),
                ExprKind::Call(func, args) => args
                    .iter()
                    .position(|arg| arg.hir_id == child_id)
                    .zip(expr_sig(cx, func))
                    .and_then(|(i, sig)| sig.input_with_hir(i))
                    .map(|(hir_ty, ty)| match hir_ty {
                        // Type inference for closures can depend on how they're called. Only go by the explicit
                        // types here.
                        Some(ty) => binding_ty_auto_deref_stability(ty),
                        None => param_auto_deref_stability(ty.skip_binder()),
                    }),
                ExprKind::MethodCall(_, args, _) => {
                    let id = cx.typeck_results().type_dependent_def_id(parent.hir_id).unwrap();
                    args.iter().position(|arg| arg.hir_id == child_id).map(|i| {
                        if i == 0 {
                            if e.hir_id == child_id {
                                Position::MethodReceiver
                            } else {
                                Position::ReborrowStable
                            }
                        } else {
                            param_auto_deref_stability(cx.tcx.fn_sig(id).skip_binder().inputs()[i])
                        }
                    })
                },
                ExprKind::Struct(path, fields, _) => {
                    let variant = variant_of_res(cx, cx.qpath_res(path, parent.hir_id));
                    fields
                        .iter()
                        .find(|f| f.expr.hir_id == child_id)
                        .zip(variant)
                        .and_then(|(field, variant)| variant.fields.iter().find(|f| f.name == field.ident.name))
                        .map(|field| param_auto_deref_stability(cx.tcx.type_of(field.did)))
                },
                ExprKind::Field(child, name) if child.hir_id == e.hir_id => Some(Position::FieldAccess(name.name)),
                ExprKind::Unary(UnOp::Deref, child) if child.hir_id == e.hir_id => Some(Position::Deref),
                ExprKind::Match(child, _, MatchSource::TryDesugar | MatchSource::AwaitDesugar)
                | ExprKind::Index(child, _)
                    if child.hir_id == e.hir_id =>
                {
                    Some(Position::Postfix)
                },
                _ => None,
            },
            _ => None,
        }
    })
    .unwrap_or(Position::Other);
    (position, adjustments)
}

// Checks the stability of auto-deref when assigned to a binding with the given explicit type.
//
// e.g.
// let x = Box::new(Box::new(0u32));
// let y1: &Box<_> = x.deref();
// let y2: &Box<_> = &x;
//
// Here `y1` and `y2` would resolve to different types, so the type `&Box<_>` is not stable when
// switching to auto-dereferencing.
fn binding_ty_auto_deref_stability(ty: &hir::Ty<'_>) -> Position {
    let TyKind::Rptr(_, ty) = &ty.kind else {
        return Position::Other;
    };
    let mut ty = ty;

    loop {
        break match ty.ty.kind {
            TyKind::Rptr(_, ref ref_ty) => {
                ty = ref_ty;
                continue;
            },
            TyKind::Path(
                QPath::TypeRelative(_, path)
                | QPath::Resolved(
                    _,
                    Path {
                        segments: [.., path], ..
                    },
                ),
            ) => {
                if let Some(args) = path.args
                    && args.args.iter().any(|arg| match arg {
                        GenericArg::Infer(_) => true,
                        GenericArg::Type(ty) => ty_contains_infer(ty),
                        _ => false,
                    })
                {
                    Position::ReborrowStable
                } else {
                    Position::DerefStable
                }
            },
            TyKind::Slice(_)
            | TyKind::Array(..)
            | TyKind::BareFn(_)
            | TyKind::Never
            | TyKind::Tup(_)
            | TyKind::Ptr(_)
            | TyKind::TraitObject(..)
            | TyKind::Path(_) => Position::DerefStable,
            TyKind::OpaqueDef(..) | TyKind::Infer | TyKind::Typeof(..) | TyKind::Err => Position::ReborrowStable,
        };
    }
}

// Checks whether a type is inferred at some point.
// e.g. `_`, `Box<_>`, `[_]`
fn ty_contains_infer(ty: &hir::Ty<'_>) -> bool {
    match &ty.kind {
        TyKind::Slice(ty) | TyKind::Array(ty, _) => ty_contains_infer(ty),
        TyKind::Ptr(ty) | TyKind::Rptr(_, ty) => ty_contains_infer(ty.ty),
        TyKind::Tup(tys) => tys.iter().any(ty_contains_infer),
        TyKind::BareFn(ty) => {
            if ty.decl.inputs.iter().any(ty_contains_infer) {
                return true;
            }
            if let FnRetTy::Return(ty) = &ty.decl.output {
                ty_contains_infer(ty)
            } else {
                false
            }
        },
        &TyKind::Path(
            QPath::TypeRelative(_, path)
            | QPath::Resolved(
                _,
                Path {
                    segments: [.., path], ..
                },
            ),
        ) => path.args.map_or(false, |args| {
            args.args.iter().any(|arg| match arg {
                GenericArg::Infer(_) => true,
                GenericArg::Type(ty) => ty_contains_infer(ty),
                _ => false,
            })
        }),
        TyKind::Path(_) | TyKind::OpaqueDef(..) | TyKind::Infer | TyKind::Typeof(_) | TyKind::Err => true,
        TyKind::Never | TyKind::TraitObject(..) => false,
    }
}

// Checks whether a type is stable when switching to auto dereferencing,
fn param_auto_deref_stability(ty: Ty<'_>) -> Position {
    let ty::Ref(_, mut ty, _) = *ty.kind() else {
        return Position::Other;
    };

    loop {
        break match *ty.kind() {
            ty::Ref(_, ref_ty, _) => {
                ty = ref_ty;
                continue;
            },
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Foreign(_)
            | ty::Str
            | ty::Array(..)
            | ty::Slice(..)
            | ty::RawPtr(..)
            | ty::FnDef(..)
            | ty::FnPtr(_)
            | ty::Closure(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
            | ty::Never
            | ty::Tuple(_)
            | ty::Projection(_) => Position::DerefStable,
            ty::Infer(_)
            | ty::Error(_)
            | ty::Param(_)
            | ty::Bound(..)
            | ty::Opaque(..)
            | ty::Placeholder(_)
            | ty::Dynamic(..) => Position::ReborrowStable,
            ty::Adt(..) if ty.has_placeholders() || ty.has_param_types_or_consts() => Position::ReborrowStable,
            ty::Adt(..) => Position::DerefStable,
        };
    }
}

fn ty_contains_field(ty: Ty<'_>, name: Symbol) -> bool {
    if let ty::Adt(adt, _) = *ty.kind() {
        adt.is_struct() && adt.non_enum_variant().fields.iter().any(|f| f.name == name)
    } else {
        false
    }
}

#[expect(clippy::needless_pass_by_value)]
fn report<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, state: State, data: StateData) {
    match state {
        State::DerefMethod {
            ty_changed_count,
            is_final_ufcs,
            target_mut,
        } => {
            let mut app = Applicability::MachineApplicable;
            let (expr_str, expr_is_macro_call) = snippet_with_context(cx, expr.span, data.span.ctxt(), "..", &mut app);
            let ty = cx.typeck_results().expr_ty(expr);
            let (_, ref_count) = peel_mid_ty_refs(ty);
            let deref_str = if ty_changed_count >= ref_count && ref_count != 0 {
                // a deref call changing &T -> &U requires two deref operators the first time
                // this occurs. One to remove the reference, a second to call the deref impl.
                "*".repeat(ty_changed_count + 1)
            } else {
                "*".repeat(ty_changed_count)
            };
            let addr_of_str = if ty_changed_count < ref_count {
                // Check if a reborrow from &mut T -> &T is required.
                if target_mut == Mutability::Not && matches!(ty.kind(), ty::Ref(_, _, Mutability::Mut)) {
                    "&*"
                } else {
                    ""
                }
            } else if target_mut == Mutability::Mut {
                "&mut "
            } else {
                "&"
            };

            let expr_str = if !expr_is_macro_call && is_final_ufcs && expr.precedence().order() < PREC_PREFIX {
                format!("({})", expr_str)
            } else {
                expr_str.into_owned()
            };

            span_lint_and_sugg(
                cx,
                EXPLICIT_DEREF_METHODS,
                data.span,
                match target_mut {
                    Mutability::Not => "explicit `deref` method call",
                    Mutability::Mut => "explicit `deref_mut` method call",
                },
                "try this",
                format!("{}{}{}", addr_of_str, deref_str, expr_str),
                app,
            );
        },
        State::DerefedBorrow(state) => {
            let mut app = Applicability::MachineApplicable;
            let snip = snippet_with_context(cx, expr.span, data.span.ctxt(), "..", &mut app).0;
            span_lint_hir_and_then(cx, NEEDLESS_BORROW, data.hir_id, data.span, state.msg, |diag| {
                let sugg = if state.required_precedence > expr.precedence().order() && !has_enclosing_paren(&snip) {
                    format!("({})", snip)
                } else {
                    snip.into()
                };
                diag.span_suggestion(data.span, "change this to", sugg, app);
            });
        },
        State::ExplicitDeref {
            deref_span,
            deref_hir_id,
        } => {
            let (span, hir_id) = if cx.typeck_results().expr_ty(expr).is_ref() {
                (data.span, data.hir_id)
            } else {
                (deref_span, deref_hir_id)
            };
            span_lint_hir_and_then(
                cx,
                EXPLICIT_AUTO_DEREF,
                hir_id,
                span,
                "deref which would be done by auto-deref",
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let snip = snippet_with_context(cx, expr.span, span.ctxt(), "..", &mut app).0;
                    diag.span_suggestion(span, "try this", snip.into_owned(), app);
                },
            );
        },
        State::ExplicitDerefField { .. } => {
            span_lint_hir_and_then(
                cx,
                EXPLICIT_AUTO_DEREF,
                data.hir_id,
                data.span,
                "deref which would be done by auto-deref",
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let snip = snippet_with_context(cx, expr.span, data.span.ctxt(), "..", &mut app).0;
                    diag.span_suggestion(data.span, "try this", snip.into_owned(), app);
                },
            );
        },
        State::Borrow | State::Reborrow { .. } => (),
    }
}

impl Dereferencing {
    fn check_local_usage<'tcx>(&mut self, cx: &LateContext<'tcx>, e: &Expr<'tcx>, local: HirId) {
        if let Some(outer_pat) = self.ref_locals.get_mut(&local) {
            if let Some(pat) = outer_pat {
                // Check for auto-deref
                if !matches!(
                    cx.typeck_results().expr_adjustments(e),
                    [
                        Adjustment {
                            kind: Adjust::Deref(_),
                            ..
                        },
                        Adjustment {
                            kind: Adjust::Deref(_),
                            ..
                        },
                        ..
                    ]
                ) {
                    match get_parent_expr(cx, e) {
                        // Field accesses are the same no matter the number of references.
                        Some(Expr {
                            kind: ExprKind::Field(..),
                            ..
                        }) => (),
                        Some(&Expr {
                            span,
                            kind: ExprKind::Unary(UnOp::Deref, _),
                            ..
                        }) if !span.from_expansion() => {
                            // Remove explicit deref.
                            let snip = snippet_with_context(cx, e.span, span.ctxt(), "..", &mut pat.app).0;
                            pat.replacements.push((span, snip.into()));
                        },
                        Some(parent) if !parent.span.from_expansion() => {
                            // Double reference might be needed at this point.
                            if parent.precedence().order() == PREC_POSTFIX {
                                // Parentheses would be needed here, don't lint.
                                *outer_pat = None;
                            } else {
                                pat.always_deref = false;
                                let snip = snippet_with_context(cx, e.span, parent.span.ctxt(), "..", &mut pat.app).0;
                                pat.replacements.push((e.span, format!("&{}", snip)));
                            }
                        },
                        _ if !e.span.from_expansion() => {
                            // Double reference might be needed at this point.
                            pat.always_deref = false;
                            let snip = snippet_with_applicability(cx, e.span, "..", &mut pat.app);
                            pat.replacements.push((e.span, format!("&{}", snip)));
                        },
                        // Edge case for macros. The span of the identifier will usually match the context of the
                        // binding, but not if the identifier was created in a macro. e.g. `concat_idents` and proc
                        // macros
                        _ => *outer_pat = None,
                    }
                }
            }
        }
    }
}
