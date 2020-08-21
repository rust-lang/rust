//! Transforms `ast::Expr` into an equivalent `hir_def::expr::Expr`
//! representation.

use std::{any::type_name, sync::Arc};

use arena::Arena;
use either::Either;
use hir_expand::{
    hygiene::Hygiene,
    name::{name, AsName, Name},
    HirFileId, MacroDefId, MacroDefKind,
};
use rustc_hash::FxHashMap;
use syntax::{
    ast::{
        self, ArgListOwner, ArrayExprKind, AstChildren, LiteralKind, LoopBodyOwner, NameOwner,
        SlicePatComponents,
    },
    AstNode, AstPtr,
};
use test_utils::mark;

use crate::{
    adt::StructKind,
    body::{Body, BodySourceMap, Expander, PatPtr, SyntheticSyntax},
    builtin_type::{BuiltinFloat, BuiltinInt},
    db::DefDatabase,
    expr::{
        dummy_expr_id, ArithOp, Array, BinaryOp, BindingAnnotation, CmpOp, Expr, ExprId, Literal,
        LogicOp, MatchArm, Ordering, Pat, PatId, RecordFieldPat, RecordLitField, Statement,
    },
    item_scope::BuiltinShadowMode,
    item_tree::{ItemTree, ItemTreeId, ItemTreeNode},
    path::{GenericArgs, Path},
    type_ref::{Mutability, Rawness, TypeRef},
    AdtId, ConstLoc, ContainerId, DefWithBodyId, EnumLoc, FunctionLoc, Intern, ModuleDefId,
    StaticLoc, StructLoc, TraitLoc, TypeAliasLoc, UnionLoc,
};

use super::{ExprSource, PatSource};

pub(crate) struct LowerCtx {
    hygiene: Hygiene,
}

impl LowerCtx {
    pub fn new(db: &dyn DefDatabase, file_id: HirFileId) -> Self {
        LowerCtx { hygiene: Hygiene::new(db.upcast(), file_id) }
    }
    pub fn with_hygiene(hygiene: &Hygiene) -> Self {
        LowerCtx { hygiene: hygiene.clone() }
    }

    pub fn lower_path(&self, ast: ast::Path) -> Option<Path> {
        Path::from_src(ast, &self.hygiene)
    }
}

pub(super) fn lower(
    db: &dyn DefDatabase,
    def: DefWithBodyId,
    expander: Expander,
    params: Option<ast::ParamList>,
    body: Option<ast::Expr>,
) -> (Body, BodySourceMap) {
    let item_tree = db.item_tree(expander.current_file_id);
    ExprCollector {
        db,
        def,
        source_map: BodySourceMap::default(),
        body: Body {
            exprs: Arena::default(),
            pats: Arena::default(),
            params: Vec::new(),
            body_expr: dummy_expr_id(),
            item_scope: Default::default(),
        },
        item_trees: {
            let mut map = FxHashMap::default();
            map.insert(expander.current_file_id, item_tree);
            map
        },
        expander,
    }
    .collect(params, body)
}

struct ExprCollector<'a> {
    db: &'a dyn DefDatabase,
    def: DefWithBodyId,
    expander: Expander,
    body: Body,
    source_map: BodySourceMap,

    item_trees: FxHashMap<HirFileId, Arc<ItemTree>>,
}

impl ExprCollector<'_> {
    fn collect(
        mut self,
        param_list: Option<ast::ParamList>,
        body: Option<ast::Expr>,
    ) -> (Body, BodySourceMap) {
        if let Some(param_list) = param_list {
            if let Some(self_param) = param_list.self_param() {
                let ptr = AstPtr::new(&self_param);
                let param_pat = self.alloc_pat(
                    Pat::Bind {
                        name: name![self],
                        mode: BindingAnnotation::Unannotated,
                        subpat: None,
                    },
                    Either::Right(ptr),
                );
                self.body.params.push(param_pat);
            }

            for param in param_list.params() {
                let pat = match param.pat() {
                    None => continue,
                    Some(pat) => pat,
                };
                let param_pat = self.collect_pat(pat);
                self.body.params.push(param_pat);
            }
        };

        self.body.body_expr = self.collect_expr_opt(body);
        (self.body, self.source_map)
    }

    fn ctx(&self) -> LowerCtx {
        LowerCtx::new(self.db, self.expander.current_file_id)
    }

    fn alloc_expr(&mut self, expr: Expr, ptr: AstPtr<ast::Expr>) -> ExprId {
        let src = self.expander.to_source(ptr);
        let id = self.make_expr(expr, Ok(src.clone()));
        self.source_map.expr_map.insert(src, id);
        id
    }
    // desugared exprs don't have ptr, that's wrong and should be fixed
    // somehow.
    fn alloc_expr_desugared(&mut self, expr: Expr) -> ExprId {
        self.make_expr(expr, Err(SyntheticSyntax))
    }
    fn empty_block(&mut self) -> ExprId {
        self.alloc_expr_desugared(Expr::Block { statements: Vec::new(), tail: None, label: None })
    }
    fn missing_expr(&mut self) -> ExprId {
        self.alloc_expr_desugared(Expr::Missing)
    }
    fn make_expr(&mut self, expr: Expr, src: Result<ExprSource, SyntheticSyntax>) -> ExprId {
        let id = self.body.exprs.alloc(expr);
        self.source_map.expr_map_back.insert(id, src);
        id
    }

    fn alloc_pat(&mut self, pat: Pat, ptr: PatPtr) -> PatId {
        let src = self.expander.to_source(ptr);
        let id = self.make_pat(pat, Ok(src.clone()));
        self.source_map.pat_map.insert(src, id);
        id
    }
    fn missing_pat(&mut self) -> PatId {
        self.make_pat(Pat::Missing, Err(SyntheticSyntax))
    }
    fn make_pat(&mut self, pat: Pat, src: Result<PatSource, SyntheticSyntax>) -> PatId {
        let id = self.body.pats.alloc(pat);
        self.source_map.pat_map_back.insert(id, src);
        id
    }

    fn collect_expr(&mut self, expr: ast::Expr) -> ExprId {
        let syntax_ptr = AstPtr::new(&expr);
        if !self.expander.is_cfg_enabled(&expr) {
            return self.missing_expr();
        }

        match expr {
            ast::Expr::IfExpr(e) => {
                let then_branch = self.collect_block_opt(e.then_branch());

                let else_branch = e.else_branch().map(|b| match b {
                    ast::ElseBranch::Block(it) => self.collect_block(it),
                    ast::ElseBranch::IfExpr(elif) => {
                        let expr: ast::Expr = ast::Expr::cast(elif.syntax().clone()).unwrap();
                        self.collect_expr(expr)
                    }
                });

                let condition = match e.condition() {
                    None => self.missing_expr(),
                    Some(condition) => match condition.pat() {
                        None => self.collect_expr_opt(condition.expr()),
                        // if let -- desugar to match
                        Some(pat) => {
                            let pat = self.collect_pat(pat);
                            let match_expr = self.collect_expr_opt(condition.expr());
                            let placeholder_pat = self.missing_pat();
                            let arms = vec![
                                MatchArm { pat, expr: then_branch, guard: None },
                                MatchArm {
                                    pat: placeholder_pat,
                                    expr: else_branch.unwrap_or_else(|| self.empty_block()),
                                    guard: None,
                                },
                            ];
                            return self
                                .alloc_expr(Expr::Match { expr: match_expr, arms }, syntax_ptr);
                        }
                    },
                };

                self.alloc_expr(Expr::If { condition, then_branch, else_branch }, syntax_ptr)
            }
            ast::Expr::EffectExpr(e) => match e.effect() {
                ast::Effect::Try(_) => {
                    let body = self.collect_block_opt(e.block_expr());
                    self.alloc_expr(Expr::TryBlock { body }, syntax_ptr)
                }
                ast::Effect::Unsafe(_) => {
                    let body = self.collect_block_opt(e.block_expr());
                    self.alloc_expr(Expr::Unsafe { body }, syntax_ptr)
                }
                // FIXME: we need to record these effects somewhere...
                ast::Effect::Label(label) => match e.block_expr() {
                    Some(block) => {
                        let res = self.collect_block(block);
                        match &mut self.body.exprs[res] {
                            Expr::Block { label: block_label, .. } => {
                                *block_label =
                                    label.lifetime_token().map(|t| Name::new_lifetime(&t))
                            }
                            _ => unreachable!(),
                        }
                        res
                    }
                    None => self.missing_expr(),
                },
                // FIXME: we need to record these effects somewhere...
                ast::Effect::Async(_) => self.collect_block_opt(e.block_expr()),
            },
            ast::Expr::BlockExpr(e) => self.collect_block(e),
            ast::Expr::LoopExpr(e) => {
                let body = self.collect_block_opt(e.loop_body());
                self.alloc_expr(
                    Expr::Loop {
                        body,
                        label: e
                            .label()
                            .and_then(|l| l.lifetime_token())
                            .map(|l| Name::new_lifetime(&l)),
                    },
                    syntax_ptr,
                )
            }
            ast::Expr::WhileExpr(e) => {
                let body = self.collect_block_opt(e.loop_body());

                let condition = match e.condition() {
                    None => self.missing_expr(),
                    Some(condition) => match condition.pat() {
                        None => self.collect_expr_opt(condition.expr()),
                        // if let -- desugar to match
                        Some(pat) => {
                            mark::hit!(infer_resolve_while_let);
                            let pat = self.collect_pat(pat);
                            let match_expr = self.collect_expr_opt(condition.expr());
                            let placeholder_pat = self.missing_pat();
                            let break_ =
                                self.alloc_expr_desugared(Expr::Break { expr: None, label: None });
                            let arms = vec![
                                MatchArm { pat, expr: body, guard: None },
                                MatchArm { pat: placeholder_pat, expr: break_, guard: None },
                            ];
                            let match_expr =
                                self.alloc_expr_desugared(Expr::Match { expr: match_expr, arms });
                            return self.alloc_expr(
                                Expr::Loop {
                                    body: match_expr,
                                    label: e
                                        .label()
                                        .and_then(|l| l.lifetime_token())
                                        .map(|l| Name::new_lifetime(&l)),
                                },
                                syntax_ptr,
                            );
                        }
                    },
                };

                self.alloc_expr(
                    Expr::While {
                        condition,
                        body,
                        label: e
                            .label()
                            .and_then(|l| l.lifetime_token())
                            .map(|l| Name::new_lifetime(&l)),
                    },
                    syntax_ptr,
                )
            }
            ast::Expr::ForExpr(e) => {
                let iterable = self.collect_expr_opt(e.iterable());
                let pat = self.collect_pat_opt(e.pat());
                let body = self.collect_block_opt(e.loop_body());
                self.alloc_expr(
                    Expr::For {
                        iterable,
                        pat,
                        body,
                        label: e
                            .label()
                            .and_then(|l| l.lifetime_token())
                            .map(|l| Name::new_lifetime(&l)),
                    },
                    syntax_ptr,
                )
            }
            ast::Expr::CallExpr(e) => {
                let callee = self.collect_expr_opt(e.expr());
                let args = if let Some(arg_list) = e.arg_list() {
                    arg_list.args().map(|e| self.collect_expr(e)).collect()
                } else {
                    Vec::new()
                };
                self.alloc_expr(Expr::Call { callee, args }, syntax_ptr)
            }
            ast::Expr::MethodCallExpr(e) => {
                let receiver = self.collect_expr_opt(e.receiver());
                let args = if let Some(arg_list) = e.arg_list() {
                    arg_list.args().map(|e| self.collect_expr(e)).collect()
                } else {
                    Vec::new()
                };
                let method_name = e.name_ref().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                let generic_args =
                    e.generic_arg_list().and_then(|it| GenericArgs::from_ast(&self.ctx(), it));
                self.alloc_expr(
                    Expr::MethodCall { receiver, method_name, args, generic_args },
                    syntax_ptr,
                )
            }
            ast::Expr::MatchExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let arms = if let Some(match_arm_list) = e.match_arm_list() {
                    match_arm_list
                        .arms()
                        .map(|arm| MatchArm {
                            pat: self.collect_pat_opt(arm.pat()),
                            expr: self.collect_expr_opt(arm.expr()),
                            guard: arm
                                .guard()
                                .and_then(|guard| guard.expr())
                                .map(|e| self.collect_expr(e)),
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                self.alloc_expr(Expr::Match { expr, arms }, syntax_ptr)
            }
            ast::Expr::PathExpr(e) => {
                let path = e
                    .path()
                    .and_then(|path| self.expander.parse_path(path))
                    .map(Expr::Path)
                    .unwrap_or(Expr::Missing);
                self.alloc_expr(path, syntax_ptr)
            }
            ast::Expr::ContinueExpr(e) => self.alloc_expr(
                Expr::Continue { label: e.lifetime_token().map(|l| Name::new_lifetime(&l)) },
                syntax_ptr,
            ),
            ast::Expr::BreakExpr(e) => {
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(
                    Expr::Break { expr, label: e.lifetime_token().map(|l| Name::new_lifetime(&l)) },
                    syntax_ptr,
                )
            }
            ast::Expr::ParenExpr(e) => {
                let inner = self.collect_expr_opt(e.expr());
                // make the paren expr point to the inner expression as well
                let src = self.expander.to_source(syntax_ptr);
                self.source_map.expr_map.insert(src, inner);
                inner
            }
            ast::Expr::ReturnExpr(e) => {
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Return { expr }, syntax_ptr)
            }
            ast::Expr::RecordExpr(e) => {
                let path = e.path().and_then(|path| self.expander.parse_path(path));
                let mut field_ptrs = Vec::new();
                let record_lit = if let Some(nfl) = e.record_expr_field_list() {
                    let fields = nfl
                        .fields()
                        .inspect(|field| field_ptrs.push(AstPtr::new(field)))
                        .filter_map(|field| {
                            if !self.expander.is_cfg_enabled(&field) {
                                return None;
                            }
                            let name = field.field_name()?.as_name();

                            Some(RecordLitField {
                                name,
                                expr: match field.expr() {
                                    Some(e) => self.collect_expr(e),
                                    None => self.missing_expr(),
                                },
                            })
                        })
                        .collect();
                    let spread = nfl.spread().map(|s| self.collect_expr(s));
                    Expr::RecordLit { path, fields, spread }
                } else {
                    Expr::RecordLit { path, fields: Vec::new(), spread: None }
                };

                let res = self.alloc_expr(record_lit, syntax_ptr);
                for (i, ptr) in field_ptrs.into_iter().enumerate() {
                    let src = self.expander.to_source(ptr);
                    self.source_map.field_map.insert((res, i), src);
                }
                res
            }
            ast::Expr::FieldExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let name = match e.field_access() {
                    Some(kind) => kind.as_name(),
                    _ => Name::missing(),
                };
                self.alloc_expr(Expr::Field { expr, name }, syntax_ptr)
            }
            ast::Expr::AwaitExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                self.alloc_expr(Expr::Await { expr }, syntax_ptr)
            }
            ast::Expr::TryExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                self.alloc_expr(Expr::Try { expr }, syntax_ptr)
            }
            ast::Expr::CastExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let type_ref = TypeRef::from_ast_opt(&self.ctx(), e.ty());
                self.alloc_expr(Expr::Cast { expr, type_ref }, syntax_ptr)
            }
            ast::Expr::RefExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let raw_tok = e.raw_token().is_some();
                let mutability = if raw_tok {
                    if e.mut_token().is_some() {
                        Mutability::Mut
                    } else if e.const_token().is_some() {
                        Mutability::Shared
                    } else {
                        unreachable!("parser only remaps to raw_token() if matching mutability token follows")
                    }
                } else {
                    Mutability::from_mutable(e.mut_token().is_some())
                };
                let rawness = Rawness::from_raw(raw_tok);
                self.alloc_expr(Expr::Ref { expr, rawness, mutability }, syntax_ptr)
            }
            ast::Expr::PrefixExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                if let Some(op) = e.op_kind() {
                    self.alloc_expr(Expr::UnaryOp { expr, op }, syntax_ptr)
                } else {
                    self.alloc_expr(Expr::Missing, syntax_ptr)
                }
            }
            ast::Expr::ClosureExpr(e) => {
                let mut args = Vec::new();
                let mut arg_types = Vec::new();
                if let Some(pl) = e.param_list() {
                    for param in pl.params() {
                        let pat = self.collect_pat_opt(param.pat());
                        let type_ref = param.ty().map(|it| TypeRef::from_ast(&self.ctx(), it));
                        args.push(pat);
                        arg_types.push(type_ref);
                    }
                }
                let ret_type =
                    e.ret_type().and_then(|r| r.ty()).map(|it| TypeRef::from_ast(&self.ctx(), it));
                let body = self.collect_expr_opt(e.body());
                self.alloc_expr(Expr::Lambda { args, arg_types, ret_type, body }, syntax_ptr)
            }
            ast::Expr::BinExpr(e) => {
                let lhs = self.collect_expr_opt(e.lhs());
                let rhs = self.collect_expr_opt(e.rhs());
                let op = e.op_kind().map(BinaryOp::from);
                self.alloc_expr(Expr::BinaryOp { lhs, rhs, op }, syntax_ptr)
            }
            ast::Expr::TupleExpr(e) => {
                let exprs = e.fields().map(|expr| self.collect_expr(expr)).collect();
                self.alloc_expr(Expr::Tuple { exprs }, syntax_ptr)
            }
            ast::Expr::BoxExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                self.alloc_expr(Expr::Box { expr }, syntax_ptr)
            }

            ast::Expr::ArrayExpr(e) => {
                let kind = e.kind();

                match kind {
                    ArrayExprKind::ElementList(e) => {
                        let exprs = e.map(|expr| self.collect_expr(expr)).collect();
                        self.alloc_expr(Expr::Array(Array::ElementList(exprs)), syntax_ptr)
                    }
                    ArrayExprKind::Repeat { initializer, repeat } => {
                        let initializer = self.collect_expr_opt(initializer);
                        let repeat = self.collect_expr_opt(repeat);
                        self.alloc_expr(
                            Expr::Array(Array::Repeat { initializer, repeat }),
                            syntax_ptr,
                        )
                    }
                }
            }

            ast::Expr::Literal(e) => self.alloc_expr(Expr::Literal(e.kind().into()), syntax_ptr),
            ast::Expr::IndexExpr(e) => {
                let base = self.collect_expr_opt(e.base());
                let index = self.collect_expr_opt(e.index());
                self.alloc_expr(Expr::Index { base, index }, syntax_ptr)
            }
            ast::Expr::RangeExpr(e) => {
                let lhs = e.start().map(|lhs| self.collect_expr(lhs));
                let rhs = e.end().map(|rhs| self.collect_expr(rhs));
                match e.op_kind() {
                    Some(range_type) => {
                        self.alloc_expr(Expr::Range { lhs, rhs, range_type }, syntax_ptr)
                    }
                    None => self.alloc_expr(Expr::Missing, syntax_ptr),
                }
            }
            ast::Expr::MacroCall(e) => {
                if let Some(name) = e.is_macro_rules().map(|it| it.as_name()) {
                    let mac = MacroDefId {
                        krate: Some(self.expander.module.krate),
                        ast_id: Some(self.expander.ast_id(&e)),
                        kind: MacroDefKind::Declarative,
                        local_inner: false,
                    };
                    self.body.item_scope.define_legacy_macro(name, mac);

                    // FIXME: do we still need to allocate this as missing ?
                    self.alloc_expr(Expr::Missing, syntax_ptr)
                } else {
                    let macro_call = self.expander.to_source(AstPtr::new(&e));
                    match self.expander.enter_expand(self.db, Some(&self.body.item_scope), e) {
                        Some((mark, expansion)) => {
                            self.source_map
                                .expansions
                                .insert(macro_call, self.expander.current_file_id);

                            let item_tree = self.db.item_tree(self.expander.current_file_id);
                            self.item_trees.insert(self.expander.current_file_id, item_tree);
                            let id = self.collect_expr(expansion);
                            self.expander.exit(self.db, mark);
                            id
                        }
                        None => self.alloc_expr(Expr::Missing, syntax_ptr),
                    }
                }
            }
        }
    }

    fn find_inner_item<N: ItemTreeNode>(&self, ast: &N::Source) -> Option<ItemTreeId<N>> {
        let id = self.expander.ast_id(ast);
        let tree = &self.item_trees[&id.file_id];

        // FIXME: This probably breaks with `use` items, since they produce multiple item tree nodes

        // Root file (non-macro).
        let item_tree_id = tree
            .all_inner_items()
            .chain(tree.top_level_items().iter().copied())
            .filter_map(|mod_item| mod_item.downcast::<N>())
            .find(|tree_id| tree[*tree_id].ast_id().upcast() == id.value.upcast())
            .or_else(|| {
                log::debug!(
                    "couldn't find inner {} item for {:?} (AST: `{}` - {:?})",
                    type_name::<N>(),
                    id,
                    ast.syntax(),
                    ast.syntax(),
                );
                None
            })?;

        Some(ItemTreeId::new(id.file_id, item_tree_id))
    }

    fn collect_expr_opt(&mut self, expr: Option<ast::Expr>) -> ExprId {
        if let Some(expr) = expr {
            self.collect_expr(expr)
        } else {
            self.missing_expr()
        }
    }

    fn collect_block(&mut self, block: ast::BlockExpr) -> ExprId {
        let syntax_node_ptr = AstPtr::new(&block.clone().into());
        self.collect_block_items(&block);
        let statements = block
            .statements()
            .filter_map(|s| {
                let stmt = match s {
                    ast::Stmt::LetStmt(stmt) => {
                        let pat = self.collect_pat_opt(stmt.pat());
                        let type_ref = stmt.ty().map(|it| TypeRef::from_ast(&self.ctx(), it));
                        let initializer = stmt.initializer().map(|e| self.collect_expr(e));
                        Statement::Let { pat, type_ref, initializer }
                    }
                    ast::Stmt::ExprStmt(stmt) => {
                        Statement::Expr(self.collect_expr_opt(stmt.expr()))
                    }
                    ast::Stmt::Item(_) => return None,
                };
                Some(stmt)
            })
            .collect();
        let tail = block.expr().map(|e| self.collect_expr(e));
        self.alloc_expr(Expr::Block { statements, tail, label: None }, syntax_node_ptr)
    }

    fn collect_block_items(&mut self, block: &ast::BlockExpr) {
        let container = ContainerId::DefWithBodyId(self.def);

        let items = block
            .statements()
            .filter_map(|stmt| match stmt {
                ast::Stmt::Item(it) => Some(it),
                ast::Stmt::LetStmt(_) | ast::Stmt::ExprStmt(_) => None,
            })
            .filter_map(|item| {
                let (def, name): (ModuleDefId, Option<ast::Name>) = match item {
                    ast::Item::Fn(def) => {
                        let id = self.find_inner_item(&def)?;
                        (
                            FunctionLoc { container: container.into(), id }.intern(self.db).into(),
                            def.name(),
                        )
                    }
                    ast::Item::TypeAlias(def) => {
                        let id = self.find_inner_item(&def)?;
                        (
                            TypeAliasLoc { container: container.into(), id }.intern(self.db).into(),
                            def.name(),
                        )
                    }
                    ast::Item::Const(def) => {
                        let id = self.find_inner_item(&def)?;
                        (
                            ConstLoc { container: container.into(), id }.intern(self.db).into(),
                            def.name(),
                        )
                    }
                    ast::Item::Static(def) => {
                        let id = self.find_inner_item(&def)?;
                        (StaticLoc { container, id }.intern(self.db).into(), def.name())
                    }
                    ast::Item::Struct(def) => {
                        let id = self.find_inner_item(&def)?;
                        (StructLoc { container, id }.intern(self.db).into(), def.name())
                    }
                    ast::Item::Enum(def) => {
                        let id = self.find_inner_item(&def)?;
                        (EnumLoc { container, id }.intern(self.db).into(), def.name())
                    }
                    ast::Item::Union(def) => {
                        let id = self.find_inner_item(&def)?;
                        (UnionLoc { container, id }.intern(self.db).into(), def.name())
                    }
                    ast::Item::Trait(def) => {
                        let id = self.find_inner_item(&def)?;
                        (TraitLoc { container, id }.intern(self.db).into(), def.name())
                    }
                    ast::Item::ExternBlock(_) => return None, // FIXME: collect from extern blocks
                    ast::Item::Impl(_)
                    | ast::Item::Use(_)
                    | ast::Item::ExternCrate(_)
                    | ast::Item::Module(_)
                    | ast::Item::MacroCall(_) => return None,
                };

                Some((def, name))
            })
            .collect::<Vec<_>>();

        for (def, name) in items {
            self.body.item_scope.define_def(def);
            if let Some(name) = name {
                let vis = crate::visibility::Visibility::Public; // FIXME determine correctly
                let has_constructor = match def {
                    ModuleDefId::AdtId(AdtId::StructId(s)) => {
                        self.db.struct_data(s).variant_data.kind() != StructKind::Record
                    }
                    _ => true,
                };
                self.body.item_scope.push_res(
                    name.as_name(),
                    crate::per_ns::PerNs::from_def(def, vis, has_constructor),
                );
            }
        }
    }

    fn collect_block_opt(&mut self, expr: Option<ast::BlockExpr>) -> ExprId {
        if let Some(block) = expr {
            self.collect_block(block)
        } else {
            self.missing_expr()
        }
    }

    fn collect_pat(&mut self, pat: ast::Pat) -> PatId {
        let pattern = match &pat {
            ast::Pat::IdentPat(bp) => {
                let name = bp.name().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                let annotation =
                    BindingAnnotation::new(bp.mut_token().is_some(), bp.ref_token().is_some());
                let subpat = bp.pat().map(|subpat| self.collect_pat(subpat));
                if annotation == BindingAnnotation::Unannotated && subpat.is_none() {
                    // This could also be a single-segment path pattern. To
                    // decide that, we need to try resolving the name.
                    let (resolved, _) = self.expander.crate_def_map.resolve_path(
                        self.db,
                        self.expander.module.local_id,
                        &name.clone().into(),
                        BuiltinShadowMode::Other,
                    );
                    match resolved.take_values() {
                        Some(ModuleDefId::ConstId(_)) => Pat::Path(name.into()),
                        Some(ModuleDefId::EnumVariantId(_)) => {
                            // this is only really valid for unit variants, but
                            // shadowing other enum variants with a pattern is
                            // an error anyway
                            Pat::Path(name.into())
                        }
                        Some(ModuleDefId::AdtId(AdtId::StructId(s)))
                            if self.db.struct_data(s).variant_data.kind() != StructKind::Record =>
                        {
                            // Funnily enough, record structs *can* be shadowed
                            // by pattern bindings (but unit or tuple structs
                            // can't).
                            Pat::Path(name.into())
                        }
                        // shadowing statics is an error as well, so we just ignore that case here
                        _ => Pat::Bind { name, mode: annotation, subpat },
                    }
                } else {
                    Pat::Bind { name, mode: annotation, subpat }
                }
            }
            ast::Pat::TupleStructPat(p) => {
                let path = p.path().and_then(|path| self.expander.parse_path(path));
                let (args, ellipsis) = self.collect_tuple_pat(p.fields());
                Pat::TupleStruct { path, args, ellipsis }
            }
            ast::Pat::RefPat(p) => {
                let pat = self.collect_pat_opt(p.pat());
                let mutability = Mutability::from_mutable(p.mut_token().is_some());
                Pat::Ref { pat, mutability }
            }
            ast::Pat::PathPat(p) => {
                let path = p.path().and_then(|path| self.expander.parse_path(path));
                path.map(Pat::Path).unwrap_or(Pat::Missing)
            }
            ast::Pat::OrPat(p) => {
                let pats = p.pats().map(|p| self.collect_pat(p)).collect();
                Pat::Or(pats)
            }
            ast::Pat::ParenPat(p) => return self.collect_pat_opt(p.pat()),
            ast::Pat::TuplePat(p) => {
                let (args, ellipsis) = self.collect_tuple_pat(p.fields());
                Pat::Tuple { args, ellipsis }
            }
            ast::Pat::WildcardPat(_) => Pat::Wild,
            ast::Pat::RecordPat(p) => {
                let path = p.path().and_then(|path| self.expander.parse_path(path));
                let args: Vec<_> = p
                    .record_pat_field_list()
                    .expect("every struct should have a field list")
                    .fields()
                    .filter_map(|f| {
                        let ast_pat = f.pat()?;
                        let pat = self.collect_pat(ast_pat);
                        let name = f.field_name()?.as_name();
                        Some(RecordFieldPat { name, pat })
                    })
                    .collect();

                let ellipsis = p
                    .record_pat_field_list()
                    .expect("every struct should have a field list")
                    .dotdot_token()
                    .is_some();

                Pat::Record { path, args, ellipsis }
            }
            ast::Pat::SlicePat(p) => {
                let SlicePatComponents { prefix, slice, suffix } = p.components();

                // FIXME properly handle `RestPat`
                Pat::Slice {
                    prefix: prefix.into_iter().map(|p| self.collect_pat(p)).collect(),
                    slice: slice.map(|p| self.collect_pat(p)),
                    suffix: suffix.into_iter().map(|p| self.collect_pat(p)).collect(),
                }
            }
            ast::Pat::LiteralPat(lit) => {
                if let Some(ast_lit) = lit.literal() {
                    let expr = Expr::Literal(ast_lit.kind().into());
                    let expr_ptr = AstPtr::new(&ast::Expr::Literal(ast_lit));
                    let expr_id = self.alloc_expr(expr, expr_ptr);
                    Pat::Lit(expr_id)
                } else {
                    Pat::Missing
                }
            }
            ast::Pat::RestPat(_) => {
                // `RestPat` requires special handling and should not be mapped
                // to a Pat. Here we are using `Pat::Missing` as a fallback for
                // when `RestPat` is mapped to `Pat`, which can easily happen
                // when the source code being analyzed has a malformed pattern
                // which includes `..` in a place where it isn't valid.

                Pat::Missing
            }
            // FIXME: implement
            ast::Pat::BoxPat(_) | ast::Pat::RangePat(_) | ast::Pat::MacroPat(_) => Pat::Missing,
        };
        let ptr = AstPtr::new(&pat);
        self.alloc_pat(pattern, Either::Left(ptr))
    }

    fn collect_pat_opt(&mut self, pat: Option<ast::Pat>) -> PatId {
        if let Some(pat) = pat {
            self.collect_pat(pat)
        } else {
            self.missing_pat()
        }
    }

    fn collect_tuple_pat(&mut self, args: AstChildren<ast::Pat>) -> (Vec<PatId>, Option<usize>) {
        // Find the location of the `..`, if there is one. Note that we do not
        // consider the possiblity of there being multiple `..` here.
        let ellipsis = args.clone().position(|p| matches!(p, ast::Pat::RestPat(_)));
        // We want to skip the `..` pattern here, since we account for it above.
        let args = args
            .filter(|p| !matches!(p, ast::Pat::RestPat(_)))
            .map(|p| self.collect_pat(p))
            .collect();

        (args, ellipsis)
    }
}

impl From<ast::BinOp> for BinaryOp {
    fn from(ast_op: ast::BinOp) -> Self {
        match ast_op {
            ast::BinOp::BooleanOr => BinaryOp::LogicOp(LogicOp::Or),
            ast::BinOp::BooleanAnd => BinaryOp::LogicOp(LogicOp::And),
            ast::BinOp::EqualityTest => BinaryOp::CmpOp(CmpOp::Eq { negated: false }),
            ast::BinOp::NegatedEqualityTest => BinaryOp::CmpOp(CmpOp::Eq { negated: true }),
            ast::BinOp::LesserEqualTest => {
                BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Less, strict: false })
            }
            ast::BinOp::GreaterEqualTest => {
                BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Greater, strict: false })
            }
            ast::BinOp::LesserTest => {
                BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Less, strict: true })
            }
            ast::BinOp::GreaterTest => {
                BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Greater, strict: true })
            }
            ast::BinOp::Addition => BinaryOp::ArithOp(ArithOp::Add),
            ast::BinOp::Multiplication => BinaryOp::ArithOp(ArithOp::Mul),
            ast::BinOp::Subtraction => BinaryOp::ArithOp(ArithOp::Sub),
            ast::BinOp::Division => BinaryOp::ArithOp(ArithOp::Div),
            ast::BinOp::Remainder => BinaryOp::ArithOp(ArithOp::Rem),
            ast::BinOp::LeftShift => BinaryOp::ArithOp(ArithOp::Shl),
            ast::BinOp::RightShift => BinaryOp::ArithOp(ArithOp::Shr),
            ast::BinOp::BitwiseXor => BinaryOp::ArithOp(ArithOp::BitXor),
            ast::BinOp::BitwiseOr => BinaryOp::ArithOp(ArithOp::BitOr),
            ast::BinOp::BitwiseAnd => BinaryOp::ArithOp(ArithOp::BitAnd),
            ast::BinOp::Assignment => BinaryOp::Assignment { op: None },
            ast::BinOp::AddAssign => BinaryOp::Assignment { op: Some(ArithOp::Add) },
            ast::BinOp::DivAssign => BinaryOp::Assignment { op: Some(ArithOp::Div) },
            ast::BinOp::MulAssign => BinaryOp::Assignment { op: Some(ArithOp::Mul) },
            ast::BinOp::RemAssign => BinaryOp::Assignment { op: Some(ArithOp::Rem) },
            ast::BinOp::ShlAssign => BinaryOp::Assignment { op: Some(ArithOp::Shl) },
            ast::BinOp::ShrAssign => BinaryOp::Assignment { op: Some(ArithOp::Shr) },
            ast::BinOp::SubAssign => BinaryOp::Assignment { op: Some(ArithOp::Sub) },
            ast::BinOp::BitOrAssign => BinaryOp::Assignment { op: Some(ArithOp::BitOr) },
            ast::BinOp::BitAndAssign => BinaryOp::Assignment { op: Some(ArithOp::BitAnd) },
            ast::BinOp::BitXorAssign => BinaryOp::Assignment { op: Some(ArithOp::BitXor) },
        }
    }
}

impl From<ast::LiteralKind> for Literal {
    fn from(ast_lit_kind: ast::LiteralKind) -> Self {
        match ast_lit_kind {
            LiteralKind::IntNumber { suffix } => {
                let known_name = suffix.and_then(|it| BuiltinInt::from_suffix(&it));

                Literal::Int(Default::default(), known_name)
            }
            LiteralKind::FloatNumber { suffix } => {
                let known_name = suffix.and_then(|it| BuiltinFloat::from_suffix(&it));

                Literal::Float(Default::default(), known_name)
            }
            LiteralKind::ByteString => Literal::ByteString(Default::default()),
            LiteralKind::String => Literal::String(Default::default()),
            LiteralKind::Byte => Literal::Int(Default::default(), Some(BuiltinInt::U8)),
            LiteralKind::Bool(val) => Literal::Bool(val),
            LiteralKind::Char => Literal::Char(Default::default()),
        }
    }
}
