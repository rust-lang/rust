use hir::intravisit::Visitor;
use rustc_hir as hir;
use rustc_index::Idx;
use rustc_middle::{
    middle::region::{BodyScopeMap, FirstStatementIndex, Scope, ScopeData, ScopeMapFacade},
    ty::TyCtxt,
};
use rustc_span::source_map;

#[derive(Debug, Copy, Clone)]
struct TemporaryLifetimeScopes {
    /// Result scope of the enclosing block
    enclosing_block_result_scope: Option<Scope>,
    /// The enclosing block scope
    enclosing_block_scope: Option<Scope>,
    /// Result scope
    result_scope: Option<Scope>,
    /// Intermediate scope
    intermediate_scope: Option<Scope>,
    /// When visiting pattern, whether a super-let declaration context is active
    is_super: bool,
    /// When visiting expression, whether the expression is at result-producing location
    is_result: bool,
}

impl TemporaryLifetimeScopes {
    fn new(scope: Scope) -> Self {
        Self {
            enclosing_block_result_scope: Some(scope),
            enclosing_block_scope: Some(scope),
            result_scope: Some(scope),
            intermediate_scope: Some(scope),
            is_super: false,
            is_result: true,
        }
    }

    fn new_static() -> Self {
        Self {
            enclosing_block_result_scope: None,
            enclosing_block_scope: None,
            result_scope: None,
            intermediate_scope: None,
            is_super: false,
            is_result: true,
        }
    }

    fn enter_terminating(&mut self, id: hir::ItemLocalId) {
        debug!(?self, "enter terminating, before");
        let scope = Scope { id, data: ScopeData::Node };
        debug!("new_temp_lifetime: terminating scope {scope:?}");
        *self = Self { is_result: false, ..Self::new(scope) };
        debug!(?self, "enter terminating, before");
    }

    fn enter_remainder(&mut self, blk_id: hir::ItemLocalId, stmt_idx: usize) {
        // `enclosing_block_result_scope` is fixed for now, because
        // the remainder sits on the resulting location of its own enclosing block.
        // This means that the remainder is itself result producing and so inherits
        // the same result scope.
        debug!(?self, "enter remainder, before");
        self.enclosing_block_scope = Some(Scope {
            id: blk_id,
            data: ScopeData::Remainder(FirstStatementIndex::new(stmt_idx)),
        });
        debug!(?self, "enter remainder, after");
    }

    fn enter_if_then(&mut self, then: hir::ItemLocalId) {
        debug!(?self, "enter if then, before");
        let scope = Scope { id: then, data: ScopeData::IfThen };
        if self.is_result {
            self.intermediate_scope = Some(scope);
        } else {
            self.result_scope = self.intermediate_scope;
            self.enclosing_block_result_scope = self.intermediate_scope;
        }
        self.enclosing_block_scope = Some(scope);
        debug!(?self, "enter if then, after");
    }

    fn enter_scrutinee(&mut self, scrutinee: hir::ItemLocalId) {
        debug!(?self, "enter scrutinee, before");
        self.result_scope = self.intermediate_scope;
        self.is_result = true;
        self.intermediate_scope = Some(Scope { id: scrutinee, data: ScopeData::Node });
        debug!(?self, "enter scrutinee, after");
    }

    fn enter_let_initializer(&mut self, classic_let_init: hir::ItemLocalId) {
        debug!(?self, "enter let init, before");
        self.result_scope = self.enclosing_block_scope;
        self.intermediate_scope = Some(Scope { id: classic_let_init, data: ScopeData::Node });
        self.is_result = true;
        // consume the `super` flag
        self.is_super = false;
        debug!(?self, "enter let init, after");
    }

    fn enter_super_let_initializer(&mut self, super_let_init: hir::ItemLocalId) {
        debug!(?self, "enter super let init, before");
        self.result_scope = self.enclosing_block_result_scope;
        self.intermediate_scope = Some(Scope { id: super_let_init, data: ScopeData::Node });
        self.is_result = true;
        // consume the `super` flag
        self.is_super = false;
        debug!(?self, "enter super let init, after");
    }

    fn enter_block(&mut self, block: hir::ItemLocalId) {
        debug!(?self, "enter block, before");
        let scope = Scope { id: block, data: ScopeData::Node };
        if self.is_result {
            self.intermediate_scope = Some(scope);
        } else {
            self.result_scope = self.intermediate_scope;
            self.enclosing_block_result_scope = self.intermediate_scope;
        }
        self.enclosing_block_scope = Some(scope);
        debug!(?self, "enter block, after");
    }

    fn enter_tail_expr(&mut self, expr: hir::ItemLocalId) {
        debug!(?self, "enter tail expr, before");
        let scope = Scope { id: expr, data: ScopeData::Node };
        self.result_scope = Some(scope);
        self.intermediate_scope = Some(scope);
        debug!(?self, "enter tail expr, after");
    }

    fn resolve_temp_lifetime(&self) -> Option<Scope> {
        if self.is_result { self.result_scope } else { self.intermediate_scope }
    }
}

struct ScopeCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    terminating_scopes: hir::ItemLocalSet,

    /// Assignment of temporary lifetimes to each expression
    expr_scope: hir::ItemLocalMap<Option<Scope>>,

    /// Assignment of scope to each variable declaration, including `super let`s.
    new_var_scope: hir::ItemLocalMap<Option<Scope>>,

    cx: Context,
}

#[derive(Clone, Copy)]
struct Context {
    temp_lifetime: TemporaryLifetimeScopes,

    var_parent: Option<Scope>,
}

impl<'tcx> ScopeCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            terminating_scopes: Default::default(),
            expr_scope: Default::default(),
            new_var_scope: Default::default(),
            cx: Context { temp_lifetime: TemporaryLifetimeScopes::new_static(), var_parent: None },
        }
    }

    fn is_data_constructor(&self, callee: &hir::Expr<'_>) -> bool {
        match callee.kind {
            hir::ExprKind::Path(hir::QPath::Resolved(
                _,
                hir::Path {
                    res:
                        hir::def::Res::Def(hir::def::DefKind::Ctor(_, _), _)
                        | hir::def::Res::SelfCtor(_),
                    ..
                },
            )) => true,
            _ => false,
        }
    }
}

impl<'tcx> ScopeCollector<'tcx> {
    fn resolve_local(
        &mut self,
        pat: Option<&'tcx hir::Pat<'tcx>>,
        init: Option<&'tcx hir::Expr<'tcx>>,
        is_super: bool,
    ) {
        let prev_temp_lifetime = self.cx.temp_lifetime;

        if let Some(expr) = init {
            let init_id = expr.hir_id.local_id;
            if is_super {
                self.cx.temp_lifetime.enter_super_let_initializer(init_id)
            } else {
                self.cx.temp_lifetime.enter_let_initializer(init_id)
            }
            self.visit_expr(expr);
        }

        if let Some(pat) = pat {
            self.cx.temp_lifetime.is_super = is_super;
            self.visit_pat(pat);
        }
        self.cx.temp_lifetime = prev_temp_lifetime;
    }
}

impl<'tcx> Visitor<'tcx> for ScopeCollector<'tcx> {
    fn visit_body(&mut self, body: &'tcx hir::Body<'tcx>) {
        let body_id = body.id();
        let owner_id = self.tcx.hir().body_owner_def_id(body_id);
        let prev_cx = self.cx;
        let outer_ts = core::mem::take(&mut self.terminating_scopes);
        self.terminating_scopes.insert(body.value.hir_id.local_id);
        let scope = Scope { id: body.value.hir_id.local_id, data: ScopeData::Arguments };
        self.cx.var_parent = Some(scope);
        self.cx.temp_lifetime = TemporaryLifetimeScopes::new(scope);
        // The arguments and `self` are parented to the fn.
        for param in body.params {
            self.visit_pat(&param.pat);
        }
        if self.tcx.hir().body_owner_kind(owner_id).is_fn_or_closure() {
            self.visit_expr(&body.value)
        } else {
            self.cx.var_parent = None;
            self.cx.temp_lifetime = TemporaryLifetimeScopes::new_static();
            self.resolve_local(None, Some(&body.value), false)
        }
        self.cx = prev_cx;
        self.terminating_scopes = outer_ts;
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        let expr_id = expr.hir_id.local_id;
        if self.terminating_scopes.contains(&expr_id) {
            self.cx.temp_lifetime.enter_terminating(expr_id)
        }

        let expr_scope = self.cx.temp_lifetime.resolve_temp_lifetime();
        debug!(?expr.hir_id.local_id, ?expr.span, ?expr_scope, "resolve temp lifetime");
        self.expr_scope.insert(expr_id, expr_scope);

        {
            let terminating_scopes = &mut self.terminating_scopes;
            let mut terminating = |id: hir::ItemLocalId| {
                terminating_scopes.insert(id);
            };
            match expr.kind {
                // Conditional or repeating scopes are always terminating
                // scopes, meaning that temporaries cannot outlive them.
                // This ensures fixed size stacks.
                hir::ExprKind::Binary(
                    source_map::Spanned { node: hir::BinOpKind::And | hir::BinOpKind::Or, .. },
                    l,
                    r,
                ) => {
                    // expr is a short circuiting operator (|| or &&). As its
                    // functionality can't be overridden by traits, it always
                    // processes bool sub-expressions. bools are Copy and thus we
                    // can drop any temporaries in evaluation (read) order
                    // (with the exception of potentially failing let expressions).
                    // We achieve this by enclosing the operands in a terminating
                    // scope, both the LHS and the RHS.

                    // We optimize this a little in the presence of chains.
                    // Chains like a && b && c get lowered to AND(AND(a, b), c).
                    // In here, b and c are RHS, while a is the only LHS operand in
                    // that chain. This holds true for longer chains as well: the
                    // leading operand is always the only LHS operand that is not a
                    // binop itself. Putting a binop like AND(a, b) into a
                    // terminating scope is not useful, thus we only put the LHS
                    // into a terminating scope if it is not a binop.

                    let terminate_lhs = match l.kind {
                        // let expressions can create temporaries that live on
                        hir::ExprKind::Let(_) => false,
                        // binops already drop their temporaries, so there is no
                        // need to put them into a terminating scope.
                        // This is purely an optimization to reduce the number of
                        // terminating scopes.
                        hir::ExprKind::Binary(
                            source_map::Spanned {
                                node: hir::BinOpKind::And | hir::BinOpKind::Or,
                                ..
                            },
                            ..,
                        ) => false,
                        // otherwise: mark it as terminating
                        _ => true,
                    };
                    if terminate_lhs {
                        terminating(l.hir_id.local_id);
                    }

                    // `Let` expressions (in a let-chain) shouldn't be terminating, as their temporaries
                    // should live beyond the immediate expression
                    if !matches!(r.kind, hir::ExprKind::Let(_)) {
                        terminating(r.hir_id.local_id);
                    }
                }
                hir::ExprKind::Loop(body, _, _, _) => {
                    terminating(body.hir_id.local_id);
                }

                hir::ExprKind::DropTemps(expr) => {
                    // `DropTemps(expr)` does not denote a conditional scope.
                    // Rather, we want to achieve the same behavior as `{ let _t = expr; _t }`.
                    terminating(expr.hir_id.local_id);
                }
                _ => {}
            }
        }

        let prev_temp_lifetime = self.cx.temp_lifetime;
        match expr.kind {
            hir::ExprKind::If(cond, then, otherwise) => {
                self.cx.temp_lifetime.enter_if_then(then.hir_id.local_id);
                self.cx.temp_lifetime.enter_scrutinee(cond.hir_id.local_id);
                self.visit_expr(cond);
                self.cx.temp_lifetime = prev_temp_lifetime;
                self.cx.temp_lifetime.is_result = true;
                self.visit_expr(then);
                if let Some(otherwise) = otherwise {
                    self.cx.temp_lifetime = prev_temp_lifetime;
                    self.cx.temp_lifetime.is_result = true;
                    self.visit_expr(otherwise)
                }
            }

            hir::ExprKind::Match(scrutinee, arms, _source) => {
                self.cx.temp_lifetime.enter_scrutinee(scrutinee.hir_id.local_id);
                self.visit_expr(scrutinee);
                for arm in arms {
                    self.cx.temp_lifetime = prev_temp_lifetime;
                    self.visit_arm(arm)
                }
            }

            hir::ExprKind::Block(block, _label) => {
                self.cx.temp_lifetime.enter_block(block.hir_id.local_id);
                self.visit_block(block)
            }

            hir::ExprKind::AddrOf(_, _, subexpr)
            | hir::ExprKind::Unary(hir::UnOp::Deref, subexpr) => {
                self.cx.temp_lifetime.is_result = true;
                self.visit_expr(subexpr)
            }

            hir::ExprKind::Struct(_path, fields, struct_base) => {
                for &hir::ExprField { expr, .. } in fields {
                    self.cx.temp_lifetime = prev_temp_lifetime;
                    self.cx.temp_lifetime.is_result = true;
                    self.visit_expr(expr);
                }
                if let Some(struct_base) = struct_base {
                    self.cx.temp_lifetime = prev_temp_lifetime;
                    // FIXME: does this make sense? It looks like so.
                    self.cx.temp_lifetime.is_result = true;
                    self.visit_expr(struct_base)
                }
            }

            hir::ExprKind::Call(callee, args) if self.is_data_constructor(callee) => {
                for expr in args {
                    self.cx.temp_lifetime = prev_temp_lifetime;
                    self.cx.temp_lifetime.is_result = true;
                    self.visit_expr(expr);
                }
            }

            hir::ExprKind::Tup(subexprs) => {
                for subexpr in subexprs {
                    self.cx.temp_lifetime = prev_temp_lifetime;
                    self.cx.temp_lifetime.is_result = true;
                    self.visit_expr(subexpr);
                }
            }

            // FIXME: we propose to change enum tuple variant constructors and struct tuple constructors
            // into proper ADT constructor, instead of calls.
            _ => {
                self.cx.temp_lifetime.is_result = false;
                self.cx.temp_lifetime.result_scope = self.cx.temp_lifetime.intermediate_scope;
                hir::intravisit::walk_expr(self, expr)
            }
        }
        self.cx.temp_lifetime = prev_temp_lifetime;
    }

    fn visit_block(&mut self, blk: &'tcx hir::Block<'tcx>) {
        let prev_cx = self.cx;
        let blk_id = blk.hir_id.local_id;
        if self.terminating_scopes.contains(&blk_id) {
            self.cx.temp_lifetime.enter_terminating(blk_id)
        }
        for (i, statement) in blk.stmts.iter().enumerate() {
            match statement.kind {
                hir::StmtKind::Local(hir::Local { els: Some(els), .. }) => {
                    // Let-else has a special lexical structure for variables.
                    // First we take a checkpoint of the current scope context here.
                    let mut prev_cx = self.cx;

                    self.cx.temp_lifetime.enter_remainder(blk_id, i);
                    self.cx.var_parent = Some(Scope {
                        id: blk_id,
                        data: ScopeData::Remainder(FirstStatementIndex::new(i)),
                    });
                    self.visit_stmt(statement);
                    // We need to back out temporarily to the last enclosing scope
                    // for the `else` block, so that even the temporaries receiving
                    // extended lifetime will be dropped inside this block.
                    // We are visiting the `else` block in this order so that
                    // the sequence of visits agrees with the order in the default
                    // `hir::intravisit` visitor.
                    core::mem::swap(&mut prev_cx, &mut self.cx);
                    let els_id = els.hir_id.local_id;
                    self.terminating_scopes.insert(els_id);
                    self.visit_block(els);
                    // From now on, we continue normally.
                    self.cx = prev_cx;
                }
                hir::StmtKind::Local(..) => {
                    // Each declaration introduces a subscope for bindings
                    // introduced by the declaration; this subscope covers a
                    // suffix of the block. Each subscope in a block has the
                    // previous subscope in the block as a parent, except for
                    // the first such subscope, which has the block itself as a
                    // parent.
                    self.cx.var_parent = Some(Scope {
                        id: blk_id,
                        data: ScopeData::Remainder(FirstStatementIndex::new(i)),
                    });
                    self.cx.temp_lifetime.enter_remainder(blk_id, i);
                    self.visit_stmt(statement)
                }
                hir::StmtKind::Item(..) => {
                    // Don't create scopes for items, since they won't be
                    // lowered to THIR and MIR.
                }
                hir::StmtKind::Expr(..) | hir::StmtKind::Semi(..) => self.visit_stmt(statement),
            }
        }
        if let Some(expr) = blk.expr {
            self.cx.temp_lifetime = prev_cx.temp_lifetime;
            self.cx.temp_lifetime.enter_tail_expr(expr.hir_id.local_id);
            self.visit_expr(expr)
        }
        self.cx = prev_cx;
    }

    fn visit_stmt(&mut self, stmt: &'tcx hir::Stmt<'tcx>) {
        let stmt_id = stmt.hir_id.local_id;
        debug!("resolve_stmt(stmt.id={:?})", stmt_id);

        let prev_cx = self.cx;

        match stmt.kind {
            hir::StmtKind::Local(local) => self.visit_local(local),
            hir::StmtKind::Item(_item) => {}
            hir::StmtKind::Expr(expression) | hir::StmtKind::Semi(expression) => {
                self.cx.temp_lifetime.enter_terminating(stmt_id);
                self.visit_expr(expression)
            }
        }

        self.cx = prev_cx;
    }

    fn visit_local(&mut self, l: &'tcx hir::Local<'tcx>) {
        self.resolve_local(Some(&l.pat), l.init, l.is_super)
    }

    fn visit_arm(&mut self, arm: &'tcx hir::Arm<'tcx>) {
        let prev_cx = self.cx;

        self.cx.var_parent = Some(Scope { id: arm.hir_id.local_id, data: ScopeData::Node });

        self.terminating_scopes.insert(arm.body.hir_id.local_id);

        // First, assert that a local declaration arising from pattern matching is
        // not super.
        self.cx.temp_lifetime.is_super = false;
        self.cx.temp_lifetime.enter_block(arm.hir_id.local_id);
        let prev_temp_lifetime = self.cx.temp_lifetime;
        self.visit_pat(arm.pat);
        if let Some(expr) = arm.guard {
            self.terminating_scopes.insert(expr.hir_id.local_id);
            self.visit_expr(expr);
            self.cx.temp_lifetime = prev_temp_lifetime;
        }
        self.cx.temp_lifetime.is_result = true;
        self.visit_expr(arm.body);

        self.cx = prev_cx;
    }

    fn visit_pat(&mut self, pat: &'tcx hir::Pat<'tcx>) {
        let pat_id = pat.hir_id.local_id;

        // If this is a binding then record the lifetime of that binding.
        if let hir::PatKind::Binding(..) = pat.kind {
            let scope = if self.cx.temp_lifetime.is_super {
                self.cx.temp_lifetime.enclosing_block_result_scope
            } else {
                self.cx.temp_lifetime.enclosing_block_scope
            };
            debug!("new_temp_lifetime: var {pat_id:?} @ {scope:?}");
            self.new_var_scope.insert(pat_id, scope);
        }

        hir::intravisit::walk_pat(self, pat)
    }
}

pub fn body_scope_map<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: hir::def_id::DefId,
) -> &'tcx ScopeMapFacade<'tcx> {
    let typeck_root_def_id = tcx.typeck_root_def_id(def_id);
    if tcx.sess.at_least_rust_2024() && tcx.features().new_temp_lifetime {
        if typeck_root_def_id != def_id {
            return tcx.body_scope_map(typeck_root_def_id);
        }
        let map = if let Some(body_id) = tcx.hir().maybe_body_owned_by(def_id.expect_local()) {
            let mut collector = ScopeCollector::new(tcx);
            collector.visit_body(tcx.hir().body(body_id));
            BodyScopeMap {
                expr_scope: collector.expr_scope,
                new_var_scope: collector.new_var_scope,
            }
        } else {
            BodyScopeMap { expr_scope: <_>::default(), new_var_scope: <_>::default() }
        };
        let map = tcx.arena.alloc(map);
        tcx.arena.alloc(ScopeMapFacade::Edition2024(map))
    } else {
        tcx.arena.alloc(ScopeMapFacade::Classical(tcx.region_scope_tree(typeck_root_def_id)))
    }
}
