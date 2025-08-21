use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_hir_and_then};
use clippy_utils::source::{snippet_with_applicability, snippet_with_context};
use clippy_utils::sugg::has_enclosing_paren;
use clippy_utils::ty::{adjust_derefs_manually_drop, implements_trait, is_manually_drop};
use clippy_utils::{
    DefinedTy, ExprUseNode, expr_use_ctxt, get_parent_expr, is_block_like, is_lint_allowed, path_to_local,
    peel_middle_ty_refs,
};
use rustc_ast::util::parser::ExprPrecedence;
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{InferKind, Visitor, VisitorExt, walk_ty};
use rustc_hir::{
    self as hir, AmbigArg, BindingMode, Body, BodyId, BorrowKind, Expr, ExprKind, HirId, MatchSource, Mutability, Node,
    Pat, PatKind, Path, QPath, TyKind, UnOp,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt, TypeckResults};
use rustc_session::impl_lint_pass;
use rustc_span::symbol::sym;
use rustc_span::{Span, Symbol};
use std::borrow::Cow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for explicit `deref()` or `deref_mut()` method calls.
    ///
    /// ### Why is this bad?
    /// Dereferencing by `&*x` or `&mut *x` is clearer and more concise,
    /// when not part of a method chain.
    ///
    /// ### Example
    /// ```no_run
    /// use std::ops::Deref;
    /// let a: &mut String = &mut String::from("foo");
    /// let b: &str = a.deref();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let a: &mut String = &mut String::from("foo");
    /// let b = &*a;
    /// ```
    ///
    /// This lint excludes all of:
    /// ```rust,ignore
    /// let _ = d.unwrap().deref();
    /// let _ = Foo::deref(&foo);
    /// let _ = <Foo as Deref>::deref(&foo);
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
    /// ### Known problems
    /// The lint cannot tell when the implementation of a trait
    /// for `&T` and `T` do different things. Removing a borrow
    /// in such a case can change the semantics of the code.
    ///
    /// ### Example
    /// ```no_run
    /// fn fun(_a: &i32) {}
    ///
    /// let x: &i32 = &&&&&&5;
    /// fun(&x);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let x = Some("");
    /// if let Some(ref x) = x {
    ///     // use `x` here
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    /// ```no_run
    /// let x = String::new();
    /// let y: &str = &*x;
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x = String::new();
    /// let y: &str = &x;
    /// ```
    #[clippy::version = "1.64.0"]
    pub EXPLICIT_AUTO_DEREF,
    complexity,
    "dereferencing when the compiler would automatically dereference"
}

impl_lint_pass!(Dereferencing<'_> => [
    EXPLICIT_DEREF_METHODS,
    NEEDLESS_BORROW,
    REF_BINDING_TO_REFERENCE,
    EXPLICIT_AUTO_DEREF,
]);

#[derive(Default)]
pub struct Dereferencing<'tcx> {
    state: Option<(State, StateData<'tcx>)>,

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

#[derive(Debug)]
struct StateData<'tcx> {
    first_expr: &'tcx Expr<'tcx>,
    adjusted_ty: Ty<'tcx>,
}

#[derive(Debug)]
struct DerefedBorrow {
    count: usize,
    msg: &'static str,
    stability: TyCoercionStability,
    for_field_access: Option<Symbol>,
}

#[derive(Debug)]
enum State {
    // Any number of deref method calls.
    DerefMethod {
        // The number of calls in a sequence which changed the referenced type
        ty_changed_count: usize,
        is_ufcs: bool,
        /// The required mutability
        mutbl: Mutability,
    },
    DerefedBorrow(DerefedBorrow),
    ExplicitDeref {
        mutability: Option<Mutability>,
    },
    ExplicitDerefField {
        name: Symbol,
        derefs_manually_drop: bool,
    },
    Reborrow {
        mutability: Mutability,
    },
    Borrow {
        mutability: Mutability,
    },
}

// A reference operation considered by this lint pass
enum RefOp {
    Method { mutbl: Mutability, is_ufcs: bool },
    Deref,
    AddrOf(Mutability),
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

impl<'tcx> LateLintPass<'tcx> for Dereferencing<'tcx> {
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
                report(cx, expr, state, data, cx.typeck_results());
            }
            return;
        }

        let typeck = cx.typeck_results();
        let Some((kind, sub_expr, skip_expr)) = try_parse_ref_op(cx.tcx, typeck, expr) else {
            // The whole chain of reference operations has been seen
            if let Some((state, data)) = self.state.take() {
                report(cx, expr, state, data, typeck);
            }
            return;
        };
        self.skip_expr = skip_expr;

        match (self.state.take(), kind) {
            (None, kind) => {
                let expr_ty = typeck.expr_ty(expr);
                let use_cx = expr_use_ctxt(cx, expr);
                let adjusted_ty = use_cx.adjustments.last().map_or(expr_ty, |a| a.target);

                match kind {
                    RefOp::Deref if use_cx.same_ctxt => {
                        let use_node = use_cx.use_node(cx);
                        let sub_ty = typeck.expr_ty(sub_expr);
                        if let ExprUseNode::FieldAccess(name) = use_node
                            && !use_cx.moved_before_use
                            && !ty_contains_field(sub_ty, name.name)
                        {
                            self.state = Some((
                                State::ExplicitDerefField {
                                    name: name.name,
                                    derefs_manually_drop: is_manually_drop(sub_ty),
                                },
                                StateData {
                                    first_expr: expr,
                                    adjusted_ty,
                                },
                            ));
                        } else if sub_ty.is_ref()
                            // Linting method receivers would require verifying that name lookup
                            // would resolve the same way. This is complicated by trait methods.
                            && !use_node.is_recv()
                            && let Some(ty) = use_node.defined_ty(cx)
                            && TyCoercionStability::for_defined_ty(cx, ty, use_node.is_return()).is_deref_stable()
                        {
                            self.state = Some((
                                State::ExplicitDeref { mutability: None },
                                StateData {
                                    first_expr: expr,
                                    adjusted_ty,
                                },
                            ));
                        }
                    },
                    RefOp::Method { mutbl, is_ufcs }
                        if !is_lint_allowed(cx, EXPLICIT_DEREF_METHODS, expr.hir_id)
                            // Allow explicit deref in method chains. e.g. `foo.deref().bar()`
                            && (is_ufcs || !is_in_method_chain(cx, expr)) =>
                    {
                        let ty_changed_count = usize::from(!deref_method_same_type(expr_ty, typeck.expr_ty(sub_expr)));
                        self.state = Some((
                            State::DerefMethod {
                                ty_changed_count,
                                is_ufcs,
                                mutbl,
                            },
                            StateData {
                                first_expr: expr,
                                adjusted_ty,
                            },
                        ));
                    },
                    RefOp::AddrOf(mutability) if use_cx.same_ctxt => {
                        // Find the number of times the borrow is auto-derefed.
                        let mut iter = use_cx.adjustments.iter();
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
                            }
                        };

                        let use_node = use_cx.use_node(cx);
                        let stability = use_node.defined_ty(cx).map_or(TyCoercionStability::None, |ty| {
                            TyCoercionStability::for_defined_ty(cx, ty, use_node.is_return())
                        });
                        let can_auto_borrow = match use_node {
                            ExprUseNode::FieldAccess(_)
                                if !use_cx.moved_before_use && matches!(sub_expr.kind, ExprKind::Field(..)) =>
                            {
                                // `DerefMut` will not be automatically applied to `ManuallyDrop<_>`
                                // field expressions when the base type is a union and the parent
                                // expression is also a field access.
                                //
                                // e.g. `&mut x.y.z` where `x` is a union, and accessing `z` requires a
                                // deref through `ManuallyDrop<_>` will not compile.
                                !adjust_derefs_manually_drop(use_cx.adjustments, expr_ty)
                            },
                            ExprUseNode::Callee | ExprUseNode::FieldAccess(_) if !use_cx.moved_before_use => true,
                            ExprUseNode::MethodArg(hir_id, _, 0) if !use_cx.moved_before_use => {
                                // Check for calls to trait methods where the trait is implemented
                                // on a reference.
                                // Two cases need to be handled:
                                // * `self` methods on `&T` will never have auto-borrow
                                // * `&self` methods on `&T` can have auto-borrow, but `&self` methods on `T` will take
                                //   priority.
                                if let Some(fn_id) = typeck.type_dependent_def_id(hir_id)
                                    && let Some(trait_id) = cx.tcx.trait_of_assoc(fn_id)
                                    && let arg_ty = cx.tcx.erase_and_anonymize_regions(adjusted_ty)
                                    && let ty::Ref(_, sub_ty, _) = *arg_ty.kind()
                                    && let args =
                                        typeck.node_args_opt(hir_id).map(|args| &args[1..]).unwrap_or_default()
                                    && let impl_ty =
                                        if cx.tcx.fn_sig(fn_id).instantiate_identity().skip_binder().inputs()[0]
                                            .is_ref()
                                        {
                                            // Trait methods taking `&self`
                                            sub_ty
                                        } else {
                                            // Trait methods taking `self`
                                            arg_ty
                                        }
                                    && impl_ty.is_ref()
                                    && implements_trait(
                                        cx,
                                        impl_ty,
                                        trait_id,
                                        &args[..cx.tcx.generics_of(trait_id).own_params.len() - 1],
                                    )
                                {
                                    false
                                } else {
                                    true
                                }
                            },
                            _ => false,
                        };

                        let deref_msg =
                            "this expression creates a reference which is immediately dereferenced by the compiler";
                        let borrow_msg = "this expression borrows a value the compiler would automatically borrow";

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
                        //    moving a mutable borrow. e.g.
                        //
                        //    ```rust
                        //    fn foo<T>(x: &mut Option<&mut T>, y: &mut T) {
                        //        let x = match x {
                        //            // Removing the borrow will cause `x` to be moved
                        //            Some(x) => &mut *x,
                        //            None => y
                        //        };
                        //    }
                        //    ```
                        let (required_refs, msg) = if can_auto_borrow {
                            (1, if deref_count == 1 { borrow_msg } else { deref_msg })
                        } else if let Some(&Adjustment {
                            kind: Adjust::Borrow(AutoBorrow::Ref(mutability)),
                            ..
                        }) = next_adjust
                            && matches!(mutability, AutoBorrowMutability::Mut { .. })
                            && !stability.is_reborrow_stable()
                        {
                            (3, deref_msg)
                        } else {
                            (2, deref_msg)
                        };

                        if deref_count >= required_refs {
                            self.state = Some((
                                State::DerefedBorrow(DerefedBorrow {
                                    // One of the required refs is for the current borrow expression, the remaining ones
                                    // can't be removed without breaking the code. See earlier comment.
                                    count: deref_count - required_refs,
                                    msg,
                                    stability,
                                    for_field_access: if let ExprUseNode::FieldAccess(name) = use_node
                                        && !use_cx.moved_before_use
                                    {
                                        Some(name.name)
                                    } else {
                                        None
                                    },
                                }),
                                StateData {
                                    first_expr: expr,
                                    adjusted_ty,
                                },
                            ));
                        } else if stability.is_deref_stable()
                            // Auto-deref doesn't combine with other adjustments
                            && next_adjust.is_none_or(|a| matches!(a.kind, Adjust::Deref(_) | Adjust::Borrow(_)))
                            && iter.all(|a| matches!(a.kind, Adjust::Deref(_) | Adjust::Borrow(_)))
                        {
                            self.state = Some((
                                State::Borrow { mutability },
                                StateData {
                                    first_expr: expr,
                                    adjusted_ty,
                                },
                            ));
                        }
                    },
                    _ => {},
                }
            },
            (
                Some((
                    State::DerefMethod {
                        mutbl,
                        ty_changed_count,
                        ..
                    },
                    data,
                )),
                RefOp::Method { is_ufcs, .. },
            ) => {
                self.state = Some((
                    State::DerefMethod {
                        ty_changed_count: if deref_method_same_type(typeck.expr_ty(expr), typeck.expr_ty(sub_expr)) {
                            ty_changed_count
                        } else {
                            ty_changed_count + 1
                        },
                        is_ufcs,
                        mutbl,
                    },
                    data,
                ));
            },
            (Some((State::DerefedBorrow(state), data)), RefOp::AddrOf(_)) if state.count != 0 => {
                self.state = Some((
                    State::DerefedBorrow(DerefedBorrow {
                        count: state.count - 1,
                        ..state
                    }),
                    data,
                ));
            },
            (Some((State::DerefedBorrow(state), data)), RefOp::AddrOf(mutability)) => {
                let adjusted_ty = data.adjusted_ty;
                let stability = state.stability;
                report(cx, expr, State::DerefedBorrow(state), data, typeck);
                if stability.is_deref_stable() {
                    self.state = Some((
                        State::Borrow { mutability },
                        StateData {
                            first_expr: expr,
                            adjusted_ty,
                        },
                    ));
                }
            },
            (Some((State::DerefedBorrow(state), data)), RefOp::Deref) => {
                let adjusted_ty = data.adjusted_ty;
                let stability = state.stability;
                let for_field_access = state.for_field_access;
                report(cx, expr, State::DerefedBorrow(state), data, typeck);
                if let Some(name) = for_field_access
                    && let sub_expr_ty = typeck.expr_ty(sub_expr)
                    && !ty_contains_field(sub_expr_ty, name)
                {
                    self.state = Some((
                        State::ExplicitDerefField {
                            name,
                            derefs_manually_drop: is_manually_drop(sub_expr_ty),
                        },
                        StateData {
                            first_expr: expr,
                            adjusted_ty,
                        },
                    ));
                } else if stability.is_deref_stable()
                    && let Some(parent) = get_parent_expr(cx, expr)
                {
                    self.state = Some((
                        State::ExplicitDeref { mutability: None },
                        StateData {
                            first_expr: parent,
                            adjusted_ty,
                        },
                    ));
                }
            },

            (Some((State::Borrow { mutability }, data)), RefOp::Deref) => {
                if typeck.expr_ty(sub_expr).is_ref() {
                    self.state = Some((State::Reborrow { mutability }, data));
                } else {
                    self.state = Some((
                        State::ExplicitDeref {
                            mutability: Some(mutability),
                        },
                        data,
                    ));
                }
            },
            (Some((State::Reborrow { mutability }, data)), RefOp::Deref) => {
                self.state = Some((
                    State::ExplicitDeref {
                        mutability: Some(mutability),
                    },
                    data,
                ));
            },
            (state @ Some((State::ExplicitDeref { .. }, _)), RefOp::Deref) => {
                self.state = state;
            },
            (
                Some((
                    State::ExplicitDerefField {
                        name,
                        derefs_manually_drop,
                    },
                    data,
                )),
                RefOp::Deref,
            ) if let sub_expr_ty = typeck.expr_ty(sub_expr)
                && !ty_contains_field(sub_expr_ty, name) =>
            {
                self.state = Some((
                    State::ExplicitDerefField {
                        name,
                        derefs_manually_drop: derefs_manually_drop || is_manually_drop(sub_expr_ty),
                    },
                    data,
                ));
            },

            (Some((state, data)), _) => report(cx, expr, state, data, typeck),
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>) {
        if let PatKind::Binding(BindingMode::REF, id, name, _) = pat.kind {
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

            if !pat.span.from_expansion()
                && let ty::Ref(_, tam, _) = *cx.typeck_results().pat_ty(pat).kind()
                // only lint immutable refs, because borrowed `&mut T` cannot be moved out
                && let ty::Ref(_, _, Mutability::Not) = *tam.kind()
            {
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
                        hir_id: pat.hir_id,
                    }),
                );
            }
        }
    }

    fn check_body_post(&mut self, cx: &LateContext<'tcx>, body: &Body<'_>) {
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
                        diag.multipart_suggestion("try", replacements, app);
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
) -> Option<(RefOp, &'tcx Expr<'tcx>, Option<HirId>)> {
    let (call_path_id, def_id, arg) = match expr.kind {
        ExprKind::MethodCall(_, arg, [], _) => (None, typeck.type_dependent_def_id(expr.hir_id)?, arg),
        ExprKind::Call(
            &Expr {
                kind: ExprKind::Path(QPath::Resolved(None, path)),
                hir_id,
                ..
            },
            [arg],
        ) => (Some(hir_id), path.res.opt_def_id()?, arg),
        ExprKind::Unary(UnOp::Deref, sub_expr) if !typeck.expr_ty(sub_expr).is_raw_ptr() => {
            return Some((RefOp::Deref, sub_expr, None));
        },
        ExprKind::AddrOf(BorrowKind::Ref, mutability, sub_expr) => {
            return Some((RefOp::AddrOf(mutability), sub_expr, None));
        },
        _ => return None,
    };
    let mutbl = match tcx.get_diagnostic_name(def_id) {
        Some(sym::deref_method) => Mutability::Not,
        Some(sym::deref_mut_method) => Mutability::Mut,
        _ => return None,
    };
    Some((
        RefOp::Method {
            mutbl,
            is_ufcs: call_path_id.is_some(),
        },
        arg,
        call_path_id,
    ))
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

fn is_in_method_chain<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'tcx>) -> bool {
    if let ExprKind::MethodCall(_, recv, _, _) = e.kind
        && matches!(recv.kind, ExprKind::MethodCall(..))
    {
        return true;
    }

    if let Some(parent) = get_parent_expr(cx, e)
        && parent.span.eq_ctxt(e.span)
    {
        match parent.kind {
            ExprKind::Call(child, _) | ExprKind::MethodCall(_, child, _, _) | ExprKind::Index(child, _, _)
                if child.hir_id == e.hir_id =>
            {
                true
            },
            ExprKind::Match(.., MatchSource::TryDesugar(_) | MatchSource::AwaitDesugar) | ExprKind::Field(_, _) => true,
            _ => false,
        }
    } else {
        false
    }
}

#[derive(Clone, Copy, Debug)]
enum TyCoercionStability {
    Deref,
    Reborrow,
    None,
}
impl TyCoercionStability {
    fn is_deref_stable(self) -> bool {
        matches!(self, Self::Deref)
    }

    fn is_reborrow_stable(self) -> bool {
        matches!(self, Self::Deref | Self::Reborrow)
    }

    fn for_defined_ty<'tcx>(cx: &LateContext<'tcx>, ty: DefinedTy<'tcx>, for_return: bool) -> Self {
        match ty {
            DefinedTy::Hir(ty) => Self::for_hir_ty(ty),
            DefinedTy::Mir { def_site_def_id, ty } => Self::for_mir_ty(
                cx.tcx,
                def_site_def_id,
                cx.tcx.instantiate_bound_regions_with_erased(ty),
                for_return,
            ),
        }
    }

    // Checks the stability of type coercions when assigned to a binding with the given explicit type.
    //
    // e.g.
    // let x = Box::new(Box::new(0u32));
    // let y1: &Box<_> = x.deref();
    // let y2: &Box<_> = &x;
    //
    // Here `y1` and `y2` would resolve to different types, so the type `&Box<_>` is not stable when
    // switching to auto-dereferencing.
    fn for_hir_ty<'tcx>(ty: &'tcx hir::Ty<'tcx>) -> Self {
        let TyKind::Ref(_, ty) = &ty.kind else {
            return Self::None;
        };
        let mut ty = ty;

        loop {
            break match ty.ty.kind {
                TyKind::Ref(_, ref ref_ty) => {
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
                            hir::GenericArg::Infer(_) => true,
                            hir::GenericArg::Type(ty) => ty_contains_infer(ty.as_unambig_ty()),
                            _ => false,
                        })
                    {
                        Self::Reborrow
                    } else {
                        Self::Deref
                    }
                },
                TyKind::Slice(_)
                | TyKind::Array(..)
                | TyKind::Ptr(_)
                | TyKind::FnPtr(_)
                | TyKind::Pat(..)
                | TyKind::Never
                | TyKind::Tup(_)
                | TyKind::Path(_) => Self::Deref,
                TyKind::OpaqueDef(..)
                | TyKind::TraitAscription(..)
                | TyKind::Infer(())
                | TyKind::Typeof(..)
                | TyKind::TraitObject(..)
                | TyKind::InferDelegation(..)
                | TyKind::Err(_) => Self::Reborrow,
                TyKind::UnsafeBinder(..) => Self::None,
            };
        }
    }

    fn for_mir_ty<'tcx>(tcx: TyCtxt<'tcx>, def_site_def_id: Option<DefId>, ty: Ty<'tcx>, for_return: bool) -> Self {
        let ty::Ref(_, mut ty, _) = *ty.kind() else {
            return Self::None;
        };

        if let Some(def_id) = def_site_def_id {
            let typing_env = ty::TypingEnv::non_body_analysis(tcx, def_id);
            ty = tcx.try_normalize_erasing_regions(typing_env, ty).unwrap_or(ty);
        }
        loop {
            break match *ty.kind() {
                ty::Ref(_, ref_ty, _) => {
                    ty = ref_ty;
                    continue;
                },
                ty::Param(_) if for_return => Self::Deref,
                ty::Alias(ty::Free | ty::Inherent, _) => unreachable!("should have been normalized away above"),
                ty::Alias(ty::Projection, _) if !for_return && ty.has_non_region_param() => Self::Reborrow,
                ty::Infer(_)
                | ty::Error(_)
                | ty::Bound(..)
                | ty::Alias(ty::Opaque, ..)
                | ty::Placeholder(_)
                | ty::Dynamic(..)
                | ty::Param(_) => Self::Reborrow,
                ty::Adt(_, args)
                    if ty.has_placeholders()
                        || ty.has_opaque_types()
                        || (!for_return && args.has_non_region_param()) =>
                {
                    Self::Reborrow
                },
                ty::Bool
                | ty::Char
                | ty::Int(_)
                | ty::Uint(_)
                | ty::Array(..)
                | ty::Pat(..)
                | ty::Float(_)
                | ty::RawPtr(..)
                | ty::FnPtr(..)
                | ty::Str
                | ty::Slice(..)
                | ty::Adt(..)
                | ty::Foreign(_)
                | ty::FnDef(..)
                | ty::Coroutine(..)
                | ty::CoroutineWitness(..)
                | ty::Closure(..)
                | ty::CoroutineClosure(..)
                | ty::Never
                | ty::Tuple(_)
                | ty::Alias(ty::Projection, _)
                | ty::UnsafeBinder(_) => Self::Deref,
            };
        }
    }
}

// Checks whether a type is inferred at some point.
// e.g. `_`, `Box<_>`, `[_]`
fn ty_contains_infer(ty: &hir::Ty<'_>) -> bool {
    struct V(bool);
    impl Visitor<'_> for V {
        fn visit_infer(&mut self, inf_id: HirId, _inf_span: Span, kind: InferKind<'_>) -> Self::Result {
            if let InferKind::Ty(_) | InferKind::Ambig(_) = kind {
                self.0 = true;
            }
            self.visit_id(inf_id);
        }

        fn visit_ty(&mut self, ty: &hir::Ty<'_, AmbigArg>) {
            if self.0 || matches!(ty.kind, TyKind::OpaqueDef(..) | TyKind::Typeof(_) | TyKind::Err(_)) {
                self.0 = true;
            } else {
                walk_ty(self, ty);
            }
        }
    }
    let mut v = V(false);
    v.visit_ty_unambig(ty);
    v.0
}

fn ty_contains_field(ty: Ty<'_>, name: Symbol) -> bool {
    if let ty::Adt(adt, _) = *ty.kind() {
        adt.is_struct() && adt.all_fields().any(|f| f.name == name)
    } else {
        false
    }
}

#[expect(clippy::needless_pass_by_value, clippy::too_many_lines)]
fn report<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    state: State,
    data: StateData<'tcx>,
    typeck: &'tcx TypeckResults<'tcx>,
) {
    match state {
        State::DerefMethod {
            ty_changed_count,
            is_ufcs,
            mutbl,
        } => {
            let mut app = Applicability::MachineApplicable;
            let (expr_str, expr_is_macro_call) =
                snippet_with_context(cx, expr.span, data.first_expr.span.ctxt(), "..", &mut app);
            let ty = typeck.expr_ty(expr);
            let (_, ref_count) = peel_middle_ty_refs(ty);
            let deref_str = if ty_changed_count >= ref_count && ref_count != 0 {
                // a deref call changing &T -> &U requires two deref operators the first time
                // this occurs. One to remove the reference, a second to call the deref impl.
                "*".repeat(ty_changed_count + 1)
            } else {
                "*".repeat(ty_changed_count)
            };
            let addr_of_str = if ty_changed_count < ref_count {
                // Check if a reborrow from &mut T -> &T is required.
                if mutbl == Mutability::Not && matches!(ty.kind(), ty::Ref(_, _, Mutability::Mut)) {
                    "&*"
                } else {
                    ""
                }
            } else if mutbl == Mutability::Mut {
                "&mut "
            } else {
                "&"
            };

            let expr_str = if !expr_is_macro_call && is_ufcs && cx.precedence(expr) < ExprPrecedence::Prefix {
                Cow::Owned(format!("({expr_str})"))
            } else {
                expr_str
            };

            span_lint_and_sugg(
                cx,
                EXPLICIT_DEREF_METHODS,
                data.first_expr.span,
                match mutbl {
                    Mutability::Not => "explicit `deref` method call",
                    Mutability::Mut => "explicit `deref_mut` method call",
                },
                "try",
                format!("{addr_of_str}{deref_str}{expr_str}"),
                app,
            );
        },
        State::DerefedBorrow(state) => {
            // Do not suggest removing a non-mandatory `&` in `&*rawptr` in an `unsafe` context,
            // as this may make rustc trigger its `dangerous_implicit_autorefs` lint.
            if let ExprKind::AddrOf(BorrowKind::Ref, _, subexpr) = data.first_expr.kind
                && let ExprKind::Unary(UnOp::Deref, subsubexpr) = subexpr.kind
                && cx.typeck_results().expr_ty_adjusted(subsubexpr).is_raw_ptr()
            {
                return;
            }

            let mut app = Applicability::MachineApplicable;
            let (snip, snip_is_macro) =
                snippet_with_context(cx, expr.span, data.first_expr.span.ctxt(), "..", &mut app);
            span_lint_hir_and_then(
                cx,
                NEEDLESS_BORROW,
                data.first_expr.hir_id,
                data.first_expr.span,
                state.msg,
                |diag| {
                    let needs_paren = match cx.tcx.parent_hir_node(data.first_expr.hir_id) {
                        Node::Expr(e) => match e.kind {
                            ExprKind::Call(callee, _) if callee.hir_id != data.first_expr.hir_id => false,
                            ExprKind::Call(..) => {
                                cx.precedence(expr) < ExprPrecedence::Unambiguous
                                    || matches!(expr.kind, ExprKind::Field(..))
                            },
                            _ => cx.precedence(expr) < cx.precedence(e),
                        },
                        _ => false,
                    };
                    let is_in_tuple = matches!(
                        get_parent_expr(cx, data.first_expr),
                        Some(Expr {
                            kind: ExprKind::Tup(..),
                            ..
                        })
                    );

                    let sugg = if !snip_is_macro && needs_paren && !has_enclosing_paren(&snip) && !is_in_tuple {
                        format!("({snip})")
                    } else {
                        snip.into()
                    };
                    diag.span_suggestion(data.first_expr.span, "change this to", sugg, app);
                },
            );
        },
        State::ExplicitDeref { mutability } => {
            if is_block_like(expr)
                && let ty::Ref(_, ty, _) = data.adjusted_ty.kind()
                && ty.is_sized(cx.tcx, cx.typing_env())
            {
                // Rustc bug: auto deref doesn't work on block expression when targeting sized types.
                return;
            }

            let ty = typeck.expr_ty(expr);

            // `&&[T; N]`, or `&&..&[T; N]` (src) cannot coerce to `&[T]` (dst).
            if let ty::Ref(_, dst, _) = data.adjusted_ty.kind()
                && dst.is_slice()
            {
                let (src, n_src_refs) = peel_middle_ty_refs(ty);
                if n_src_refs >= 2 && src.is_array() {
                    return;
                }
            }

            let (prefix, needs_paren) = match mutability {
                Some(mutability) if !ty.is_ref() => {
                    let prefix = match mutability {
                        Mutability::Not => "&",
                        Mutability::Mut => "&mut ",
                    };
                    (prefix, cx.precedence(expr) < ExprPrecedence::Prefix)
                },
                None if !ty.is_ref() && data.adjusted_ty.is_ref() => ("&", false),
                _ => ("", false),
            };
            span_lint_hir_and_then(
                cx,
                EXPLICIT_AUTO_DEREF,
                data.first_expr.hir_id,
                data.first_expr.span,
                "deref which would be done by auto-deref",
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let (snip, snip_is_macro) =
                        snippet_with_context(cx, expr.span, data.first_expr.span.ctxt(), "..", &mut app);
                    let sugg = if !snip_is_macro && needs_paren && !has_enclosing_paren(&snip) {
                        format!("{prefix}({snip})")
                    } else {
                        format!("{prefix}{snip}")
                    };
                    diag.span_suggestion(data.first_expr.span, "try", sugg, app);
                },
            );
        },
        State::ExplicitDerefField {
            derefs_manually_drop, ..
        } => {
            let (snip_span, needs_parens) = if matches!(expr.kind, ExprKind::Field(..))
                && (derefs_manually_drop
                    || adjust_derefs_manually_drop(
                        typeck.expr_adjustments(data.first_expr),
                        typeck.expr_ty(data.first_expr),
                    )) {
                // `DerefMut` will not be automatically applied to `ManuallyDrop<_>`
                // field expressions when the base type is a union and the parent
                // expression is also a field access.
                //
                // e.g. `&mut x.y.z` where `x` is a union, and accessing `z` requires a
                // deref through `ManuallyDrop<_>` will not compile.
                let parent_id = cx.tcx.parent_hir_id(expr.hir_id);
                if parent_id == data.first_expr.hir_id {
                    return;
                }
                (cx.tcx.hir_node(parent_id).expect_expr().span, true)
            } else {
                (expr.span, false)
            };
            span_lint_hir_and_then(
                cx,
                EXPLICIT_AUTO_DEREF,
                data.first_expr.hir_id,
                data.first_expr.span,
                "deref which would be done by auto-deref",
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let snip = snippet_with_context(cx, snip_span, data.first_expr.span.ctxt(), "..", &mut app).0;
                    let sugg = if needs_parens {
                        format!("({snip})")
                    } else {
                        snip.into_owned()
                    };
                    diag.span_suggestion(data.first_expr.span, "try", sugg, app);
                },
            );
        },
        State::Borrow { .. } | State::Reborrow { .. } => (),
    }
}

impl<'tcx> Dereferencing<'tcx> {
    fn check_local_usage(&mut self, cx: &LateContext<'tcx>, e: &Expr<'tcx>, local: HirId) {
        if let Some(outer_pat) = self.ref_locals.get_mut(&local)
            && let Some(pat) = outer_pat
            // Check for auto-deref
            && !matches!(
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
            )
        {
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
                    if cx.precedence(parent) == ExprPrecedence::Unambiguous {
                        // Parentheses would be needed here, don't lint.
                        *outer_pat = None;
                    } else {
                        pat.always_deref = false;
                        let snip = snippet_with_context(cx, e.span, parent.span.ctxt(), "..", &mut pat.app).0;
                        pat.replacements.push((e.span, format!("&{snip}")));
                    }
                },
                _ if !e.span.from_expansion() => {
                    // Double reference might be needed at this point.
                    pat.always_deref = false;
                    let snip = snippet_with_applicability(cx, e.span, "..", &mut pat.app);
                    pat.replacements.push((e.span, format!("&{snip}")));
                },
                // Edge case for macros. The span of the identifier will usually match the context of the
                // binding, but not if the identifier was created in a macro. e.g. `concat_idents` and proc
                // macros
                _ => *outer_pat = None,
            }
        }
    }
}
