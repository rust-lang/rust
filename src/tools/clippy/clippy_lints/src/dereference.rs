use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::{snippet_with_applicability, snippet_with_context};
use clippy_utils::sugg::has_enclosing_paren;
use clippy_utils::ty::peel_mid_ty_refs;
use clippy_utils::{get_parent_expr, get_parent_node, is_lint_allowed, path_to_local};
use rustc_ast::util::parser::{PREC_POSTFIX, PREC_PREFIX};
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::Applicability;
use rustc_hir::{
    BindingAnnotation, Body, BodyId, BorrowKind, Destination, Expr, ExprKind, HirId, MatchSource, Mutability, Node,
    Pat, PatKind, UnOp,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeckResults};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{symbol::sym, Span};

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
    /// Could be written as:
    /// ```rust
    /// let a: &mut String = &mut String::from("foo");
    /// let b = &*a;
    /// ```
    ///
    /// This lint excludes
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
    /// // Bad
    /// let x: &i32 = &&&&&&5;
    /// fun(&x);
    ///
    /// // Good
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
    /// // Bad
    /// let x = Some("");
    /// if let Some(ref x) = x {
    ///     // use `x` here
    /// }
    ///
    /// // Good
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

impl_lint_pass!(Dereferencing => [
    EXPLICIT_DEREF_METHODS,
    NEEDLESS_BORROW,
    REF_BINDING_TO_REFERENCE,
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
    DerefedBorrow {
        count: usize,
        required_precedence: i8,
        msg: &'static str,
    },
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
}

impl<'tcx> LateLintPass<'tcx> for Dereferencing {
    #[allow(clippy::too_many_lines)]
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
                let parent = get_parent_node(cx.tcx, expr.hir_id);
                let expr_ty = typeck.expr_ty(expr);

                match kind {
                    RefOp::Method(target_mut)
                        if !is_lint_allowed(cx, EXPLICIT_DEREF_METHODS, expr.hir_id)
                            && is_linted_explicit_deref_position(parent, expr.hir_id, expr.span) =>
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
                            StateData { span: expr.span },
                        ));
                    },
                    RefOp::AddrOf => {
                        // Find the number of times the borrow is auto-derefed.
                        let mut iter = find_adjustments(cx.tcx, typeck, expr).iter();
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

                        let (required_refs, required_precedence, msg) = if is_auto_borrow_position(parent, expr.hir_id)
                        {
                            (1, PREC_POSTFIX, if deref_count == 1 { borrow_msg } else { deref_msg })
                        } else if let Some(&Adjust::Borrow(AutoBorrow::Ref(_, mutability))) =
                            next_adjust.map(|a| &a.kind)
                        {
                            if matches!(mutability, AutoBorrowMutability::Mut { .. })
                                && !is_auto_reborrow_position(parent)
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
                                State::DerefedBorrow {
                                    // One of the required refs is for the current borrow expression, the remaining ones
                                    // can't be removed without breaking the code. See earlier comment.
                                    count: deref_count - required_refs,
                                    required_precedence,
                                    msg,
                                },
                                StateData { span: expr.span },
                            ));
                        }
                    },
                    _ => (),
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
            (
                Some((
                    State::DerefedBorrow {
                        count,
                        required_precedence,
                        msg,
                    },
                    data,
                )),
                RefOp::AddrOf,
            ) if count != 0 => {
                self.state = Some((
                    State::DerefedBorrow {
                        count: count - 1,
                        required_precedence,
                        msg,
                    },
                    data,
                ));
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
                span_lint_and_then(
                    cx,
                    if pat.always_deref {
                        NEEDLESS_BORROW
                    } else {
                        REF_BINDING_TO_REFERENCE
                    },
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
fn deref_method_same_type(result_ty: Ty<'_>, arg_ty: Ty<'_>) -> bool {
    match (result_ty.kind(), arg_ty.kind()) {
        (ty::Ref(_, result_ty, _), ty::Ref(_, arg_ty, _)) => result_ty == arg_ty,

        // The result type for a deref method is always a reference
        // Not matching the previous pattern means the argument type is not a reference
        // This means that the type did change
        _ => false,
    }
}

// Checks whether the parent node is a suitable context for switching from a deref method to the
// deref operator.
fn is_linted_explicit_deref_position(parent: Option<Node<'_>>, child_id: HirId, child_span: Span) -> bool {
    let parent = match parent {
        Some(Node::Expr(e)) if e.span.ctxt() == child_span.ctxt() => e,
        _ => return true,
    };
    match parent.kind {
        // Leave deref calls in the middle of a method chain.
        // e.g. x.deref().foo()
        ExprKind::MethodCall(_, [self_arg, ..], _) if self_arg.hir_id == child_id => false,

        // Leave deref calls resulting in a called function
        // e.g. (x.deref())()
        ExprKind::Call(func_expr, _) if func_expr.hir_id == child_id => false,

        // Makes an ugly suggestion
        // e.g. *x.deref() => *&*x
        ExprKind::Unary(UnOp::Deref, _)
        // Postfix expressions would require parens
        | ExprKind::Match(_, _, MatchSource::TryDesugar | MatchSource::AwaitDesugar)
        | ExprKind::Field(..)
        | ExprKind::Index(..)
        | ExprKind::Err => false,

        ExprKind::Box(..)
        | ExprKind::ConstBlock(..)
        | ExprKind::Array(_)
        | ExprKind::Call(..)
        | ExprKind::MethodCall(..)
        | ExprKind::Tup(..)
        | ExprKind::Binary(..)
        | ExprKind::Unary(..)
        | ExprKind::Lit(..)
        | ExprKind::Cast(..)
        | ExprKind::Type(..)
        | ExprKind::DropTemps(..)
        | ExprKind::If(..)
        | ExprKind::Loop(..)
        | ExprKind::Match(..)
        | ExprKind::Let(..)
        | ExprKind::Closure(..)
        | ExprKind::Block(..)
        | ExprKind::Assign(..)
        | ExprKind::AssignOp(..)
        | ExprKind::Path(..)
        | ExprKind::AddrOf(..)
        | ExprKind::Break(..)
        | ExprKind::Continue(..)
        | ExprKind::Ret(..)
        | ExprKind::InlineAsm(..)
        | ExprKind::Struct(..)
        | ExprKind::Repeat(..)
        | ExprKind::Yield(..) => true,
    }
}

/// Checks if the given expression is in a position which can be auto-reborrowed.
/// Note: This is only correct assuming auto-deref is already occurring.
fn is_auto_reborrow_position(parent: Option<Node<'_>>) -> bool {
    match parent {
        Some(Node::Expr(parent)) => matches!(parent.kind, ExprKind::MethodCall(..) | ExprKind::Call(..)),
        Some(Node::Local(_)) => true,
        _ => false,
    }
}

/// Checks if the given expression is a position which can auto-borrow.
fn is_auto_borrow_position(parent: Option<Node<'_>>, child_id: HirId) -> bool {
    if let Some(Node::Expr(parent)) = parent {
        match parent.kind {
            // ExprKind::MethodCall(_, [self_arg, ..], _) => self_arg.hir_id == child_id,
            ExprKind::Field(..) => true,
            ExprKind::Call(f, _) => f.hir_id == child_id,
            _ => false,
        }
    } else {
        false
    }
}

/// Adjustments are sometimes made in the parent block rather than the expression itself.
fn find_adjustments<'tcx>(
    tcx: TyCtxt<'tcx>,
    typeck: &'tcx TypeckResults<'_>,
    expr: &'tcx Expr<'_>,
) -> &'tcx [Adjustment<'tcx>] {
    let map = tcx.hir();
    let mut iter = map.parent_iter(expr.hir_id);
    let mut prev = expr;

    loop {
        match typeck.expr_adjustments(prev) {
            [] => (),
            a => break a,
        };

        match iter.next().map(|(_, x)| x) {
            Some(Node::Block(_)) => {
                if let Some((_, Node::Expr(e))) = iter.next() {
                    prev = e;
                } else {
                    // This shouldn't happen. Blocks are always contained in an expression.
                    break &[];
                }
            },
            Some(Node::Expr(&Expr {
                kind: ExprKind::Break(Destination { target_id: Ok(id), .. }, _),
                ..
            })) => {
                if let Some(Node::Expr(e)) = map.find(id) {
                    prev = e;
                    iter = map.parent_iter(id);
                } else {
                    // This shouldn't happen. The destination should exist.
                    break &[];
                }
            },
            _ => break &[],
        }
    }
}

#[allow(clippy::needless_pass_by_value)]
fn report(cx: &LateContext<'_>, expr: &Expr<'_>, state: State, data: StateData) {
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
        State::DerefedBorrow {
            required_precedence,
            msg,
            ..
        } => {
            let mut app = Applicability::MachineApplicable;
            let snip = snippet_with_context(cx, expr.span, data.span.ctxt(), "..", &mut app).0;
            span_lint_and_sugg(
                cx,
                NEEDLESS_BORROW,
                data.span,
                msg,
                "change this to",
                if required_precedence > expr.precedence().order() && !has_enclosing_paren(&snip) {
                    format!("({})", snip)
                } else {
                    snip.into()
                },
                app,
            );
        },
    }
}

impl Dereferencing {
    fn check_local_usage(&mut self, cx: &LateContext<'_>, e: &Expr<'_>, local: HirId) {
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
