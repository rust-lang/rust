use rustc_ast::node_id::NodeMap;
use rustc_ast::*;
use rustc_hir as hir;
use rustc_hir::{HirId, Target, find_attr};
use rustc_middle::span_bug;
use rustc_span::Span;

use super::{LoweringContext, MoveExprInitializerFinder, MoveExprState};
use crate::FnDeclKind;
use crate::errors::{ClosureCannotBeStatic, CoroutineTooManyParameters};

impl<'hir> LoweringContext<'_, 'hir> {
    // Entry point for `ExprKind::Closure`. Plain closures go through
    // `lower_expr_plain_closure_with_move_exprs`, which can wrap the lowered
    // closure in `let` initializers for `move(...)`. Coroutine closures keep the
    // existing coroutine-specific path and reject `move(...)` for now.
    pub(super) fn lower_expr_closure_expr(
        &mut self,
        e: &Expr,
        closure: &Closure,
    ) -> hir::Expr<'hir> {
        let expr_hir_id = self.lower_node_id(e.id);
        let attrs = self.lower_attrs(expr_hir_id, &e.attrs, e.span, Target::from_expr(e));

        match closure.coroutine_kind {
            // FIXME(TaKO8Ki): Support `move(expr)` in coroutine closures too.
            // For the first step, we only support plain closures.
            Some(coroutine_kind) => hir::Expr {
                hir_id: expr_hir_id,
                kind: self.lower_expr_coroutine_closure(
                    &closure.binder,
                    closure.capture_clause,
                    e.id,
                    expr_hir_id,
                    coroutine_kind,
                    closure.constness,
                    &closure.fn_decl,
                    &closure.body,
                    closure.fn_decl_span,
                    closure.fn_arg_span,
                ),
                span: self.lower_span(e.span),
            },
            None => self.lower_expr_plain_closure_with_move_exprs(
                expr_hir_id,
                attrs,
                &closure.binder,
                closure.capture_clause,
                e.id,
                closure.constness,
                closure.movability,
                &closure.fn_decl,
                &closure.body,
                closure.fn_decl_span,
                closure.fn_arg_span,
                e.span,
            ),
        }
    }

    /// Lowers a plain closure expression and wraps it in an outer block if the
    /// closure body used `move(...)`.
    ///
    /// The lowering is split this way because `move(...)` initializers must be
    /// evaluated before the closure is created, but the closure body must still
    /// lower each `move(...)` occurrence as a use of the synthetic local that
    /// will be introduced by that outer block. For example:
    ///
    /// ```ignore (illustrative)
    /// || (move(move(foo.clone()))).len()
    /// ```
    ///
    /// first lowers the closure body roughly as `|| __move_expr_1.len()` while
    /// recording two occurrences:
    ///
    /// ```ignore (illustrative)
    /// move(foo.clone()) -> __move_expr_0
    /// move(move(foo.clone())) -> __move_expr_1
    /// ```
    ///
    /// This method then lowers the recorded initializers in order and builds the
    /// surrounding block:
    ///
    /// ```ignore (illustrative)
    /// {
    ///     let __move_expr_0 = foo.clone();
    ///     let __move_expr_1 = __move_expr_0;
    ///     || __move_expr_1.len()
    /// }
    /// ```
    fn lower_expr_plain_closure_with_move_exprs(
        &mut self,
        expr_hir_id: HirId,
        attrs: &[hir::Attribute],
        binder: &ClosureBinder,
        capture_clause: CaptureBy,
        closure_id: NodeId,
        constness: Const,
        movability: Movability,
        decl: &FnDecl,
        body: &Expr,
        fn_decl_span: Span,
        fn_arg_span: Span,
        whole_span: Span,
    ) -> hir::Expr<'hir> {
        let (closure_kind, move_expr_state) = self.lower_expr_closure(
            attrs,
            binder,
            capture_clause,
            closure_id,
            constness,
            movability,
            decl,
            body,
            fn_decl_span,
            fn_arg_span,
        );

        if move_expr_state.occurrences.is_empty() {
            return hir::Expr {
                hir_id: expr_hir_id,
                kind: closure_kind,
                span: self.lower_span(whole_span),
            };
        }

        let initializers = MoveExprInitializerFinder::collect(body)
            .into_iter()
            .map(|initializer| (initializer.id, initializer.expr))
            .collect::<NodeMap<_>>();
        let mut stmts = Vec::with_capacity(move_expr_state.occurrences.len());
        let mut initializer_bindings = NodeMap::default();
        for occurrence in &move_expr_state.occurrences {
            // Evaluate the expression inside `move(...)` before creating the
            // closure and store it in a synthetic local:
            // `|| move(foo).bar` becomes roughly
            // `let __move_expr_0 = foo; || __move_expr_0.bar`.
            let expr = initializers[&occurrence.id];
            let init = if initializer_bindings.is_empty() {
                self.lower_expr(expr)
            } else {
                // Earlier entries cover nested `move(...)` expressions that
                // appear inside this initializer, as in
                // `move(move(foo.clone()))`.
                let (init, _) = self.with_move_expr_bindings(
                    Some(MoveExprState {
                        bindings: initializer_bindings.clone(),
                        occurrences: Vec::new(),
                    }),
                    |this| this.lower_expr(expr),
                );
                init
            };
            stmts.push(self.stmt_let_pat(
                None,
                expr.span,
                Some(init),
                occurrence.pat,
                hir::LocalSource::Normal,
            ));
            initializer_bindings.insert(occurrence.id, (occurrence.ident, occurrence.binding));
        }

        let closure_expr = self.arena.alloc(hir::Expr {
            hir_id: expr_hir_id,
            kind: closure_kind,
            span: self.lower_span(whole_span),
        });

        let stmts = self.arena.alloc_from_iter(stmts);
        let block = self.block_all(whole_span, stmts, Some(closure_expr));
        self.expr(whole_span, hir::ExprKind::Block(block, None))
    }

    // Lowers the actual plain closure node and body. The body is lowered while a
    // `MoveExprState` is active, so `move(...)` occurrences become synthetic
    // local uses and the caller can later add the matching initializers.
    fn lower_expr_closure(
        &mut self,
        attrs: &[hir::Attribute],
        binder: &ClosureBinder,
        capture_clause: CaptureBy,
        closure_id: NodeId,
        constness: Const,
        movability: Movability,
        decl: &FnDecl,
        body: &Expr,
        fn_decl_span: Span,
        fn_arg_span: Span,
    ) -> (hir::ExprKind<'hir>, MoveExprState<'hir>) {
        let closure_def_id = self.local_def_id(closure_id);
        let (binder_clause, generic_params) = self.lower_closure_binder(binder);

        let ((body_id, closure_kind), move_expr_state) =
            self.with_new_scopes(fn_decl_span, move |this| {
                let mut coroutine_kind = find_attr!(
                    attrs,
                    Coroutine => hir::CoroutineKind::Coroutine(Movability::Movable)
                );

                this.with_move_expr_bindings(Some(MoveExprState::default()), |this| {
                    // FIXME(contracts): Support contracts on closures?
                    let body_id = this.lower_fn_body(decl, None, |this| {
                        this.coroutine_kind = coroutine_kind;
                        let e = this.lower_expr_mut(body);
                        coroutine_kind = this.coroutine_kind;
                        e
                    });
                    let coroutine_option = this.closure_movability_for_fn(
                        decl,
                        fn_decl_span,
                        coroutine_kind,
                        movability,
                    );
                    (body_id, coroutine_option)
                })
            });
        let Some(move_expr_state) = move_expr_state else {
            span_bug!(fn_decl_span, "plain closure lowering did not return `move(...)` state");
        };
        let explicit_captures: &'hir [hir::ExplicitCapture] = self.arena.alloc_from_iter(
            move_expr_state.occurrences.iter().filter_map(|occurrence| {
                occurrence
                    .explicit_capture
                    .then_some(hir::ExplicitCapture { var_hir_id: occurrence.binding })
            }),
        );

        let bound_generic_params = self.lower_lifetime_binder(closure_id, generic_params);
        // Lower outside new scope to preserve `is_in_loop_condition`.
        let fn_decl = self.lower_fn_decl(decl, closure_id, fn_decl_span, FnDeclKind::Closure, None);

        let c = self.arena.alloc(hir::Closure {
            def_id: closure_def_id,
            binder: binder_clause,
            capture_clause: self.lower_capture_clause(capture_clause),
            bound_generic_params,
            fn_decl,
            body: body_id,
            fn_decl_span: self.lower_span(fn_decl_span),
            fn_arg_span: Some(self.lower_span(fn_arg_span)),
            kind: closure_kind,
            constness: self.lower_constness(constness),
            explicit_captures,
        });

        (hir::ExprKind::Closure(c), move_expr_state)
    }

    fn closure_movability_for_fn(
        &mut self,
        decl: &FnDecl,
        fn_decl_span: Span,
        coroutine_kind: Option<hir::CoroutineKind>,
        movability: Movability,
    ) -> hir::ClosureKind {
        match coroutine_kind {
            Some(hir::CoroutineKind::Coroutine(_)) => {
                if decl.inputs.len() > 1 {
                    self.dcx().emit_err(CoroutineTooManyParameters { fn_decl_span });
                }
                hir::ClosureKind::Coroutine(hir::CoroutineKind::Coroutine(movability))
            }
            Some(
                hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen, _)
                | hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _)
                | hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::AsyncGen, _),
            ) => {
                panic!("non-`async`/`gen` closure body turned `async`/`gen` during lowering");
            }
            None => {
                if movability == Movability::Static {
                    self.dcx().emit_err(ClosureCannotBeStatic { fn_decl_span });
                }
                hir::ClosureKind::Closure
            }
        }
    }

    fn lower_closure_binder<'c>(
        &mut self,
        binder: &'c ClosureBinder,
    ) -> (hir::ClosureBinder, &'c [GenericParam]) {
        let (binder, params) = match binder {
            ClosureBinder::NotPresent => (hir::ClosureBinder::Default, &[][..]),
            ClosureBinder::For { span, generic_params } => {
                let span = self.lower_span(*span);
                (hir::ClosureBinder::For { span }, &**generic_params)
            }
        };

        (binder, params)
    }

    // Coroutine closures are lowered separately because they build a different
    // body shape. This path pushes `None` for `move_expr_bindings`, so any
    // `move(...)` in the coroutine body gets a targeted unsupported-position
    // error instead of being collected like a plain closure occurrence.
    fn lower_expr_coroutine_closure(
        &mut self,
        binder: &ClosureBinder,
        capture_clause: CaptureBy,
        closure_id: NodeId,
        closure_hir_id: HirId,
        coroutine_kind: CoroutineKind,
        constness: Const,
        decl: &FnDecl,
        body: &Expr,
        fn_decl_span: Span,
        fn_arg_span: Span,
    ) -> hir::ExprKind<'hir> {
        let closure_def_id = self.local_def_id(closure_id);
        let (binder_clause, generic_params) = self.lower_closure_binder(binder);

        let coroutine_desugaring = match coroutine_kind {
            CoroutineKind::Async { .. } => hir::CoroutineDesugaring::Async,
            CoroutineKind::Gen { .. } => hir::CoroutineDesugaring::Gen,
            CoroutineKind::AsyncGen { span, .. } => {
                span_bug!(span, "only async closures and `iter!` closures are supported currently")
            }
        };

        let body = self.with_new_scopes(fn_decl_span, |this| {
            let inner_decl =
                FnDecl { inputs: decl.inputs.clone(), output: FnRetTy::Default(fn_decl_span) };

            // Transform `async |x: u8| -> X { ... }` into
            // `|x: u8| || -> X { ... }`.
            let body_id = this.lower_body(|this| {
                let ((parameters, expr), _) = this.with_move_expr_bindings(None, |this| {
                    this.lower_coroutine_body_with_moved_arguments(
                        &inner_decl,
                        |this| this.with_new_scopes(fn_decl_span, |this| this.lower_expr_mut(body)),
                        fn_decl_span,
                        body.span,
                        coroutine_kind,
                        hir::CoroutineSource::Closure,
                    )
                });

                this.maybe_forward_track_caller(body.span, closure_hir_id, expr.hir_id);

                (parameters, expr)
            });
            body_id
        });

        let bound_generic_params = self.lower_lifetime_binder(closure_id, generic_params);
        // We need to lower the declaration outside the new scope, because we
        // have to conserve the state of being inside a loop condition for the
        // closure argument types.
        let fn_decl =
            self.lower_fn_decl(&decl, closure_id, fn_decl_span, FnDeclKind::Closure, None);

        if let Const::Yes(span) = constness {
            self.dcx().span_err(span, "const coroutines are not supported");
        }

        let c = self.arena.alloc(hir::Closure {
            def_id: closure_def_id,
            binder: binder_clause,
            capture_clause: self.lower_capture_clause(capture_clause),
            bound_generic_params,
            fn_decl,
            body,
            fn_decl_span: self.lower_span(fn_decl_span),
            fn_arg_span: Some(self.lower_span(fn_arg_span)),
            // Lower this as a `CoroutineClosure`. That will ensure that HIR typeck
            // knows that a `FnDecl` output type like `-> &str` actually means
            // "coroutine that returns &str", rather than directly returning a `&str`.
            kind: hir::ClosureKind::CoroutineClosure(coroutine_desugaring),
            constness: self.lower_constness(constness),
            explicit_captures: &[],
        });
        hir::ExprKind::Closure(c)
    }
}
