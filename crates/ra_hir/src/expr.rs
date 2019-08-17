use std::ops::Index;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use ra_arena::{impl_arena_id, map::ArenaMap, Arena, RawId};
use ra_syntax::{
    ast::{
        self, ArgListOwner, ArrayExprKind, LiteralKind, LoopBodyOwner, NameOwner,
        TryBlockBodyOwner, TypeAscriptionOwner,
    },
    AstNode, AstPtr, SyntaxNodePtr,
};
use test_utils::tested_by;

use crate::{
    name::{AsName, SELF_PARAM},
    path::GenericArgs,
    ty::primitive::{FloatTy, IntTy, UncertainFloatTy, UncertainIntTy},
    type_ref::{Mutability, TypeRef},
    DefWithBody, Either, HasSource, HirDatabase, HirFileId, MacroCallLoc, MacroFileKind, Name,
    Path, Resolver,
};

pub use self::scope::ExprScopes;

pub(crate) mod scope;
pub(crate) mod validation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprId(RawId);
impl_arena_id!(ExprId);

/// The body of an item (function, const etc.).
#[derive(Debug, Eq, PartialEq)]
pub struct Body {
    /// The def of the item this body belongs to
    owner: DefWithBody,
    exprs: Arena<ExprId, Expr>,
    pats: Arena<PatId, Pat>,
    /// The patterns for the function's parameters. While the parameter types are
    /// part of the function signature, the patterns are not (they don't change
    /// the external type of the function).
    ///
    /// If this `Body` is for the body of a constant, this will just be
    /// empty.
    params: Vec<PatId>,
    /// The `ExprId` of the actual body expression.
    body_expr: ExprId,
}

/// An item body together with the mapping from syntax nodes to HIR expression
/// IDs. This is needed to go from e.g. a position in a file to the HIR
/// expression containing it; but for type inference etc., we want to operate on
/// a structure that is agnostic to the actual positions of expressions in the
/// file, so that we don't recompute types whenever some whitespace is typed.
#[derive(Default, Debug, Eq, PartialEq)]
pub struct BodySourceMap {
    expr_map: FxHashMap<SyntaxNodePtr, ExprId>,
    expr_map_back: ArenaMap<ExprId, SyntaxNodePtr>,
    pat_map: FxHashMap<PatPtr, PatId>,
    pat_map_back: ArenaMap<PatId, PatPtr>,
    field_map: FxHashMap<(ExprId, usize), AstPtr<ast::NamedField>>,
}

type PatPtr = Either<AstPtr<ast::Pat>, AstPtr<ast::SelfParam>>;

impl Body {
    pub fn params(&self) -> &[PatId] {
        &self.params
    }

    pub fn body_expr(&self) -> ExprId {
        self.body_expr
    }

    pub fn owner(&self) -> DefWithBody {
        self.owner
    }

    pub fn exprs(&self) -> impl Iterator<Item = (ExprId, &Expr)> {
        self.exprs.iter()
    }

    pub fn pats(&self) -> impl Iterator<Item = (PatId, &Pat)> {
        self.pats.iter()
    }
}

// needs arbitrary_self_types to be a method... or maybe move to the def?
pub(crate) fn resolver_for_expr(
    body: Arc<Body>,
    db: &impl HirDatabase,
    expr_id: ExprId,
) -> Resolver {
    let scopes = db.expr_scopes(body.owner);
    resolver_for_scope(body, db, scopes.scope_for(expr_id))
}

pub(crate) fn resolver_for_scope(
    body: Arc<Body>,
    db: &impl HirDatabase,
    scope_id: Option<scope::ScopeId>,
) -> Resolver {
    let mut r = body.owner.resolver(db);
    let scopes = db.expr_scopes(body.owner);
    let scope_chain = scopes.scope_chain(scope_id).collect::<Vec<_>>();
    for scope in scope_chain.into_iter().rev() {
        r = r.push_expr_scope(Arc::clone(&scopes), scope);
    }
    r
}

impl Index<ExprId> for Body {
    type Output = Expr;

    fn index(&self, expr: ExprId) -> &Expr {
        &self.exprs[expr]
    }
}

impl Index<PatId> for Body {
    type Output = Pat;

    fn index(&self, pat: PatId) -> &Pat {
        &self.pats[pat]
    }
}

impl BodySourceMap {
    pub(crate) fn expr_syntax(&self, expr: ExprId) -> Option<SyntaxNodePtr> {
        self.expr_map_back.get(expr).cloned()
    }

    pub(crate) fn syntax_expr(&self, ptr: SyntaxNodePtr) -> Option<ExprId> {
        self.expr_map.get(&ptr).cloned()
    }

    pub(crate) fn node_expr(&self, node: &ast::Expr) -> Option<ExprId> {
        self.expr_map.get(&SyntaxNodePtr::new(node.syntax())).cloned()
    }

    pub(crate) fn pat_syntax(&self, pat: PatId) -> Option<PatPtr> {
        self.pat_map_back.get(pat).cloned()
    }

    pub(crate) fn node_pat(&self, node: &ast::Pat) -> Option<PatId> {
        self.pat_map.get(&Either::A(AstPtr::new(node))).cloned()
    }

    pub(crate) fn field_syntax(&self, expr: ExprId, field: usize) -> AstPtr<ast::NamedField> {
        self.field_map[&(expr, field)]
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Literal {
    String(String),
    ByteString(Vec<u8>),
    Char(char),
    Bool(bool),
    Int(u64, UncertainIntTy),
    Float(u64, UncertainFloatTy), // FIXME: f64 is not Eq
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Expr {
    /// This is produced if syntax tree does not have a required expression piece.
    Missing,
    Path(Path),
    If {
        condition: ExprId,
        then_branch: ExprId,
        else_branch: Option<ExprId>,
    },
    Block {
        statements: Vec<Statement>,
        tail: Option<ExprId>,
    },
    Loop {
        body: ExprId,
    },
    While {
        condition: ExprId,
        body: ExprId,
    },
    For {
        iterable: ExprId,
        pat: PatId,
        body: ExprId,
    },
    Call {
        callee: ExprId,
        args: Vec<ExprId>,
    },
    MethodCall {
        receiver: ExprId,
        method_name: Name,
        args: Vec<ExprId>,
        generic_args: Option<GenericArgs>,
    },
    Match {
        expr: ExprId,
        arms: Vec<MatchArm>,
    },
    Continue,
    Break {
        expr: Option<ExprId>,
    },
    Return {
        expr: Option<ExprId>,
    },
    StructLit {
        path: Option<Path>,
        fields: Vec<StructLitField>,
        spread: Option<ExprId>,
    },
    Field {
        expr: ExprId,
        name: Name,
    },
    Await {
        expr: ExprId,
    },
    Try {
        expr: ExprId,
    },
    TryBlock {
        body: ExprId,
    },
    Cast {
        expr: ExprId,
        type_ref: TypeRef,
    },
    Ref {
        expr: ExprId,
        mutability: Mutability,
    },
    UnaryOp {
        expr: ExprId,
        op: UnaryOp,
    },
    BinaryOp {
        lhs: ExprId,
        rhs: ExprId,
        op: Option<BinaryOp>,
    },
    Index {
        base: ExprId,
        index: ExprId,
    },
    Lambda {
        args: Vec<PatId>,
        arg_types: Vec<Option<TypeRef>>,
        body: ExprId,
    },
    Tuple {
        exprs: Vec<ExprId>,
    },
    Array(Array),
    Literal(Literal),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    LogicOp(LogicOp),
    ArithOp(ArithOp),
    CmpOp(CmpOp),
    Assignment { op: Option<ArithOp> },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LogicOp {
    And,
    Or,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq { negated: bool },
    Ord { ordering: Ordering, strict: bool },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ordering {
    Less,
    Greater,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ArithOp {
    Add,
    Mul,
    Sub,
    Div,
    Rem,
    Shl,
    Shr,
    BitXor,
    BitOr,
    BitAnd,
}

pub use ra_syntax::ast::PrefixOp as UnaryOp;
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Array {
    ElementList(Vec<ExprId>),
    Repeat { initializer: ExprId, repeat: ExprId },
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MatchArm {
    pub pats: Vec<PatId>,
    pub guard: Option<ExprId>,
    pub expr: ExprId,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StructLitField {
    pub name: Name,
    pub expr: ExprId,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Statement {
    Let { pat: PatId, type_ref: Option<TypeRef>, initializer: Option<ExprId> },
    Expr(ExprId),
}

impl Expr {
    pub fn walk_child_exprs(&self, mut f: impl FnMut(ExprId)) {
        match self {
            Expr::Missing => {}
            Expr::Path(_) => {}
            Expr::If { condition, then_branch, else_branch } => {
                f(*condition);
                f(*then_branch);
                if let Some(else_branch) = else_branch {
                    f(*else_branch);
                }
            }
            Expr::Block { statements, tail } => {
                for stmt in statements {
                    match stmt {
                        Statement::Let { initializer, .. } => {
                            if let Some(expr) = initializer {
                                f(*expr);
                            }
                        }
                        Statement::Expr(e) => f(*e),
                    }
                }
                if let Some(expr) = tail {
                    f(*expr);
                }
            }
            Expr::TryBlock { body } => f(*body),
            Expr::Loop { body } => f(*body),
            Expr::While { condition, body } => {
                f(*condition);
                f(*body);
            }
            Expr::For { iterable, body, .. } => {
                f(*iterable);
                f(*body);
            }
            Expr::Call { callee, args } => {
                f(*callee);
                for arg in args {
                    f(*arg);
                }
            }
            Expr::MethodCall { receiver, args, .. } => {
                f(*receiver);
                for arg in args {
                    f(*arg);
                }
            }
            Expr::Match { expr, arms } => {
                f(*expr);
                for arm in arms {
                    f(arm.expr);
                }
            }
            Expr::Continue => {}
            Expr::Break { expr } | Expr::Return { expr } => {
                if let Some(expr) = expr {
                    f(*expr);
                }
            }
            Expr::StructLit { fields, spread, .. } => {
                for field in fields {
                    f(field.expr);
                }
                if let Some(expr) = spread {
                    f(*expr);
                }
            }
            Expr::Lambda { body, .. } => {
                f(*body);
            }
            Expr::BinaryOp { lhs, rhs, .. } => {
                f(*lhs);
                f(*rhs);
            }
            Expr::Index { base, index } => {
                f(*base);
                f(*index);
            }
            Expr::Field { expr, .. }
            | Expr::Await { expr }
            | Expr::Try { expr }
            | Expr::Cast { expr, .. }
            | Expr::Ref { expr, .. }
            | Expr::UnaryOp { expr, .. } => {
                f(*expr);
            }
            Expr::Tuple { exprs } => {
                for expr in exprs {
                    f(*expr);
                }
            }
            Expr::Array(a) => match a {
                Array::ElementList(exprs) => {
                    for expr in exprs {
                        f(*expr);
                    }
                }
                Array::Repeat { initializer, repeat } => {
                    f(*initializer);
                    f(*repeat)
                }
            },
            Expr::Literal(_) => {}
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PatId(RawId);
impl_arena_id!(PatId);

/// Explicit binding annotations given in the HIR for a binding. Note
/// that this is not the final binding *mode* that we infer after type
/// inference.
#[derive(Clone, PartialEq, Eq, Debug, Copy)]
pub enum BindingAnnotation {
    /// No binding annotation given: this means that the final binding mode
    /// will depend on whether we have skipped through a `&` reference
    /// when matching. For example, the `x` in `Some(x)` will have binding
    /// mode `None`; if you do `let Some(x) = &Some(22)`, it will
    /// ultimately be inferred to be by-reference.
    Unannotated,

    /// Annotated with `mut x` -- could be either ref or not, similar to `None`.
    Mutable,

    /// Annotated as `ref`, like `ref x`
    Ref,

    /// Annotated as `ref mut x`.
    RefMut,
}

impl BindingAnnotation {
    fn new(is_mutable: bool, is_ref: bool) -> Self {
        match (is_mutable, is_ref) {
            (true, true) => BindingAnnotation::RefMut,
            (false, true) => BindingAnnotation::Ref,
            (true, false) => BindingAnnotation::Mutable,
            (false, false) => BindingAnnotation::Unannotated,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FieldPat {
    pub(crate) name: Name,
    pub(crate) pat: PatId,
}

/// Close relative to rustc's hir::PatKind
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Pat {
    Missing,
    Wild,
    Tuple(Vec<PatId>),
    Struct {
        path: Option<Path>,
        args: Vec<FieldPat>,
        // FIXME: 'ellipsis' option
    },
    Range {
        start: ExprId,
        end: ExprId,
    },
    Slice {
        prefix: Vec<PatId>,
        rest: Option<PatId>,
        suffix: Vec<PatId>,
    },
    Path(Path),
    Lit(ExprId),
    Bind {
        mode: BindingAnnotation,
        name: Name,
        subpat: Option<PatId>,
    },
    TupleStruct {
        path: Option<Path>,
        args: Vec<PatId>,
    },
    Ref {
        pat: PatId,
        mutability: Mutability,
    },
}

impl Pat {
    pub fn walk_child_pats(&self, mut f: impl FnMut(PatId)) {
        match self {
            Pat::Range { .. } | Pat::Lit(..) | Pat::Path(..) | Pat::Wild | Pat::Missing => {}
            Pat::Bind { subpat, .. } => {
                subpat.iter().copied().for_each(f);
            }
            Pat::Tuple(args) | Pat::TupleStruct { args, .. } => {
                args.iter().copied().for_each(f);
            }
            Pat::Ref { pat, .. } => f(*pat),
            Pat::Slice { prefix, rest, suffix } => {
                let total_iter = prefix.iter().chain(rest.iter()).chain(suffix.iter());
                total_iter.copied().for_each(f);
            }
            Pat::Struct { args, .. } => {
                args.iter().map(|f| f.pat).for_each(f);
            }
        }
    }
}

// Queries

pub(crate) struct ExprCollector<DB> {
    db: DB,
    owner: DefWithBody,
    exprs: Arena<ExprId, Expr>,
    pats: Arena<PatId, Pat>,
    source_map: BodySourceMap,
    params: Vec<PatId>,
    body_expr: Option<ExprId>,
    resolver: Resolver,
    // Expr collector expands macros along the way. original points to the file
    // we started with, current points to the current macro expansion. source
    // maps don't support macros yet, so we only record info into source map if
    // current == original (see #1196)
    original_file_id: HirFileId,
    current_file_id: HirFileId,
}

impl<'a, DB> ExprCollector<&'a DB>
where
    DB: HirDatabase,
{
    fn new(owner: DefWithBody, file_id: HirFileId, resolver: Resolver, db: &'a DB) -> Self {
        ExprCollector {
            owner,
            resolver,
            db,
            exprs: Arena::default(),
            pats: Arena::default(),
            source_map: BodySourceMap::default(),
            params: Vec::new(),
            body_expr: None,
            original_file_id: file_id,
            current_file_id: file_id,
        }
    }
    fn alloc_expr(&mut self, expr: Expr, syntax_ptr: SyntaxNodePtr) -> ExprId {
        let id = self.exprs.alloc(expr);
        if self.current_file_id == self.original_file_id {
            self.source_map.expr_map.insert(syntax_ptr, id);
            self.source_map.expr_map_back.insert(id, syntax_ptr);
        }
        id
    }

    fn alloc_pat(&mut self, pat: Pat, ptr: PatPtr) -> PatId {
        let id = self.pats.alloc(pat);

        if self.current_file_id == self.original_file_id {
            self.source_map.pat_map.insert(ptr, id);
            self.source_map.pat_map_back.insert(id, ptr);
        }

        id
    }

    fn empty_block(&mut self) -> ExprId {
        let block = Expr::Block { statements: Vec::new(), tail: None };
        self.exprs.alloc(block)
    }

    fn collect_expr(&mut self, expr: ast::Expr) -> ExprId {
        let syntax_ptr = SyntaxNodePtr::new(expr.syntax());
        match expr.kind() {
            ast::ExprKind::IfExpr(e) => {
                let then_branch = self.collect_block_opt(e.then_branch());

                let else_branch = e.else_branch().map(|b| match b {
                    ast::ElseBranch::Block(it) => self.collect_block(it),
                    ast::ElseBranch::IfExpr(elif) => {
                        let expr: ast::Expr = ast::Expr::cast(elif.syntax().clone()).unwrap();
                        self.collect_expr(expr)
                    }
                });

                let condition = match e.condition() {
                    None => self.exprs.alloc(Expr::Missing),
                    Some(condition) => match condition.pat() {
                        None => self.collect_expr_opt(condition.expr()),
                        // if let -- desugar to match
                        Some(pat) => {
                            let pat = self.collect_pat(pat);
                            let match_expr = self.collect_expr_opt(condition.expr());
                            let placeholder_pat = self.pats.alloc(Pat::Missing);
                            let arms = vec![
                                MatchArm { pats: vec![pat], expr: then_branch, guard: None },
                                MatchArm {
                                    pats: vec![placeholder_pat],
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
            ast::ExprKind::TryBlockExpr(e) => {
                let body = self.collect_block_opt(e.try_body());
                self.alloc_expr(Expr::TryBlock { body }, syntax_ptr)
            }
            ast::ExprKind::BlockExpr(e) => self.collect_block_opt(e.block()),
            ast::ExprKind::LoopExpr(e) => {
                let body = self.collect_block_opt(e.loop_body());
                self.alloc_expr(Expr::Loop { body }, syntax_ptr)
            }
            ast::ExprKind::WhileExpr(e) => {
                let body = self.collect_block_opt(e.loop_body());

                let condition = match e.condition() {
                    None => self.exprs.alloc(Expr::Missing),
                    Some(condition) => match condition.pat() {
                        None => self.collect_expr_opt(condition.expr()),
                        // if let -- desugar to match
                        Some(pat) => {
                            tested_by!(infer_while_let);
                            let pat = self.collect_pat(pat);
                            let match_expr = self.collect_expr_opt(condition.expr());
                            let placeholder_pat = self.pats.alloc(Pat::Missing);
                            let break_ = self.exprs.alloc(Expr::Break { expr: None });
                            let arms = vec![
                                MatchArm { pats: vec![pat], expr: body, guard: None },
                                MatchArm { pats: vec![placeholder_pat], expr: break_, guard: None },
                            ];
                            let match_expr =
                                self.exprs.alloc(Expr::Match { expr: match_expr, arms });
                            return self.alloc_expr(Expr::Loop { body: match_expr }, syntax_ptr);
                        }
                    },
                };

                self.alloc_expr(Expr::While { condition, body }, syntax_ptr)
            }
            ast::ExprKind::ForExpr(e) => {
                let iterable = self.collect_expr_opt(e.iterable());
                let pat = self.collect_pat_opt(e.pat());
                let body = self.collect_block_opt(e.loop_body());
                self.alloc_expr(Expr::For { iterable, pat, body }, syntax_ptr)
            }
            ast::ExprKind::CallExpr(e) => {
                let callee = self.collect_expr_opt(e.expr());
                let args = if let Some(arg_list) = e.arg_list() {
                    arg_list.args().map(|e| self.collect_expr(e)).collect()
                } else {
                    Vec::new()
                };
                self.alloc_expr(Expr::Call { callee, args }, syntax_ptr)
            }
            ast::ExprKind::MethodCallExpr(e) => {
                let receiver = self.collect_expr_opt(e.expr());
                let args = if let Some(arg_list) = e.arg_list() {
                    arg_list.args().map(|e| self.collect_expr(e)).collect()
                } else {
                    Vec::new()
                };
                let method_name = e.name_ref().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                let generic_args = e.type_arg_list().and_then(GenericArgs::from_ast);
                self.alloc_expr(
                    Expr::MethodCall { receiver, method_name, args, generic_args },
                    syntax_ptr,
                )
            }
            ast::ExprKind::MatchExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let arms = if let Some(match_arm_list) = e.match_arm_list() {
                    match_arm_list
                        .arms()
                        .map(|arm| MatchArm {
                            pats: arm.pats().map(|p| self.collect_pat(p)).collect(),
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
            ast::ExprKind::PathExpr(e) => {
                let path =
                    e.path().and_then(Path::from_ast).map(Expr::Path).unwrap_or(Expr::Missing);
                self.alloc_expr(path, syntax_ptr)
            }
            ast::ExprKind::ContinueExpr(_e) => {
                // FIXME: labels
                self.alloc_expr(Expr::Continue, syntax_ptr)
            }
            ast::ExprKind::BreakExpr(e) => {
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Break { expr }, syntax_ptr)
            }
            ast::ExprKind::ParenExpr(e) => {
                let inner = self.collect_expr_opt(e.expr());
                // make the paren expr point to the inner expression as well
                self.source_map.expr_map.insert(syntax_ptr, inner);
                inner
            }
            ast::ExprKind::ReturnExpr(e) => {
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Return { expr }, syntax_ptr)
            }
            ast::ExprKind::StructLit(e) => {
                let path = e.path().and_then(Path::from_ast);
                let mut field_ptrs = Vec::new();
                let struct_lit = if let Some(nfl) = e.named_field_list() {
                    let fields = nfl
                        .fields()
                        .inspect(|field| field_ptrs.push(AstPtr::new(field)))
                        .map(|field| StructLitField {
                            name: field
                                .name_ref()
                                .map(|nr| nr.as_name())
                                .unwrap_or_else(Name::missing),
                            expr: if let Some(e) = field.expr() {
                                self.collect_expr(e)
                            } else if let Some(nr) = field.name_ref() {
                                // field shorthand
                                let id = self.exprs.alloc(Expr::Path(Path::from_name_ref(&nr)));
                                self.source_map
                                    .expr_map
                                    .insert(SyntaxNodePtr::new(nr.syntax()), id);
                                self.source_map
                                    .expr_map_back
                                    .insert(id, SyntaxNodePtr::new(nr.syntax()));
                                id
                            } else {
                                self.exprs.alloc(Expr::Missing)
                            },
                        })
                        .collect();
                    let spread = nfl.spread().map(|s| self.collect_expr(s));
                    Expr::StructLit { path, fields, spread }
                } else {
                    Expr::StructLit { path, fields: Vec::new(), spread: None }
                };

                let res = self.alloc_expr(struct_lit, syntax_ptr);
                for (i, ptr) in field_ptrs.into_iter().enumerate() {
                    self.source_map.field_map.insert((res, i), ptr);
                }
                res
            }
            ast::ExprKind::FieldExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let name = match e.field_access() {
                    Some(kind) => kind.as_name(),
                    _ => Name::missing(),
                };
                self.alloc_expr(Expr::Field { expr, name }, syntax_ptr)
            }
            ast::ExprKind::AwaitExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                self.alloc_expr(Expr::Await { expr }, syntax_ptr)
            }
            ast::ExprKind::TryExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                self.alloc_expr(Expr::Try { expr }, syntax_ptr)
            }
            ast::ExprKind::CastExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let type_ref = TypeRef::from_ast_opt(e.type_ref());
                self.alloc_expr(Expr::Cast { expr, type_ref }, syntax_ptr)
            }
            ast::ExprKind::RefExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let mutability = Mutability::from_mutable(e.is_mut());
                self.alloc_expr(Expr::Ref { expr, mutability }, syntax_ptr)
            }
            ast::ExprKind::PrefixExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                if let Some(op) = e.op_kind() {
                    self.alloc_expr(Expr::UnaryOp { expr, op }, syntax_ptr)
                } else {
                    self.alloc_expr(Expr::Missing, syntax_ptr)
                }
            }
            ast::ExprKind::LambdaExpr(e) => {
                let mut args = Vec::new();
                let mut arg_types = Vec::new();
                if let Some(pl) = e.param_list() {
                    for param in pl.params() {
                        let pat = self.collect_pat_opt(param.pat());
                        let type_ref = param.ascribed_type().map(TypeRef::from_ast);
                        args.push(pat);
                        arg_types.push(type_ref);
                    }
                }
                let body = self.collect_expr_opt(e.body());
                self.alloc_expr(Expr::Lambda { args, arg_types, body }, syntax_ptr)
            }
            ast::ExprKind::BinExpr(e) => {
                let lhs = self.collect_expr_opt(e.lhs());
                let rhs = self.collect_expr_opt(e.rhs());
                let op = e.op_kind().map(BinaryOp::from);
                self.alloc_expr(Expr::BinaryOp { lhs, rhs, op }, syntax_ptr)
            }
            ast::ExprKind::TupleExpr(e) => {
                let exprs = e.exprs().map(|expr| self.collect_expr(expr)).collect();
                self.alloc_expr(Expr::Tuple { exprs }, syntax_ptr)
            }

            ast::ExprKind::ArrayExpr(e) => {
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

            ast::ExprKind::Literal(e) => {
                let lit = match e.kind() {
                    LiteralKind::IntNumber { suffix } => {
                        let known_name = suffix
                            .and_then(|it| IntTy::from_suffix(&it).map(UncertainIntTy::Known));

                        Literal::Int(
                            Default::default(),
                            known_name.unwrap_or(UncertainIntTy::Unknown),
                        )
                    }
                    LiteralKind::FloatNumber { suffix } => {
                        let known_name = suffix
                            .and_then(|it| FloatTy::from_suffix(&it).map(UncertainFloatTy::Known));

                        Literal::Float(
                            Default::default(),
                            known_name.unwrap_or(UncertainFloatTy::Unknown),
                        )
                    }
                    LiteralKind::ByteString => Literal::ByteString(Default::default()),
                    LiteralKind::String => Literal::String(Default::default()),
                    LiteralKind::Byte => {
                        Literal::Int(Default::default(), UncertainIntTy::Known(IntTy::u8()))
                    }
                    LiteralKind::Bool => Literal::Bool(Default::default()),
                    LiteralKind::Char => Literal::Char(Default::default()),
                };
                self.alloc_expr(Expr::Literal(lit), syntax_ptr)
            }
            ast::ExprKind::IndexExpr(e) => {
                let base = self.collect_expr_opt(e.base());
                let index = self.collect_expr_opt(e.index());
                self.alloc_expr(Expr::Index { base, index }, syntax_ptr)
            }

            // FIXME implement HIR for these:
            ast::ExprKind::Label(_e) => self.alloc_expr(Expr::Missing, syntax_ptr),
            ast::ExprKind::RangeExpr(_e) => self.alloc_expr(Expr::Missing, syntax_ptr),
            ast::ExprKind::MacroCall(e) => {
                let ast_id = self
                    .db
                    .ast_id_map(self.current_file_id)
                    .ast_id(&e)
                    .with_file_id(self.current_file_id);

                if let Some(path) = e.path().and_then(Path::from_ast) {
                    if let Some(def) = self.resolver.resolve_path_as_macro(self.db, &path) {
                        let call_id = MacroCallLoc { def: def.id, ast_id }.id(self.db);
                        let file_id = call_id.as_file(MacroFileKind::Expr);
                        if let Some(node) = self.db.parse_or_expand(file_id) {
                            if let Some(expr) = ast::Expr::cast(node) {
                                log::debug!("macro expansion {:#?}", expr.syntax());
                                let old_file_id =
                                    std::mem::replace(&mut self.current_file_id, file_id);
                                let id = self.collect_expr(expr);
                                self.current_file_id = old_file_id;
                                return id;
                            }
                        }
                    }
                }
                // FIXME: Instead of just dropping the error from expansion
                // report it
                self.alloc_expr(Expr::Missing, syntax_ptr)
            }
        }
    }

    fn collect_expr_opt(&mut self, expr: Option<ast::Expr>) -> ExprId {
        if let Some(expr) = expr {
            self.collect_expr(expr)
        } else {
            self.exprs.alloc(Expr::Missing)
        }
    }

    fn collect_block(&mut self, block: ast::Block) -> ExprId {
        let statements = block
            .statements()
            .map(|s| match s.kind() {
                ast::StmtKind::LetStmt(stmt) => {
                    let pat = self.collect_pat_opt(stmt.pat());
                    let type_ref = stmt.ascribed_type().map(TypeRef::from_ast);
                    let initializer = stmt.initializer().map(|e| self.collect_expr(e));
                    Statement::Let { pat, type_ref, initializer }
                }
                ast::StmtKind::ExprStmt(stmt) => {
                    Statement::Expr(self.collect_expr_opt(stmt.expr()))
                }
            })
            .collect();
        let tail = block.expr().map(|e| self.collect_expr(e));
        self.alloc_expr(Expr::Block { statements, tail }, SyntaxNodePtr::new(block.syntax()))
    }

    fn collect_block_opt(&mut self, block: Option<ast::Block>) -> ExprId {
        if let Some(block) = block {
            self.collect_block(block)
        } else {
            self.exprs.alloc(Expr::Missing)
        }
    }

    fn collect_pat(&mut self, pat: ast::Pat) -> PatId {
        let pattern = match pat.kind() {
            ast::PatKind::BindPat(bp) => {
                let name = bp.name().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                let annotation = BindingAnnotation::new(bp.is_mutable(), bp.is_ref());
                let subpat = bp.pat().map(|subpat| self.collect_pat(subpat));
                Pat::Bind { name, mode: annotation, subpat }
            }
            ast::PatKind::TupleStructPat(p) => {
                let path = p.path().and_then(Path::from_ast);
                let args = p.args().map(|p| self.collect_pat(p)).collect();
                Pat::TupleStruct { path, args }
            }
            ast::PatKind::RefPat(p) => {
                let pat = self.collect_pat_opt(p.pat());
                let mutability = Mutability::from_mutable(p.is_mut());
                Pat::Ref { pat, mutability }
            }
            ast::PatKind::PathPat(p) => {
                let path = p.path().and_then(Path::from_ast);
                path.map(Pat::Path).unwrap_or(Pat::Missing)
            }
            ast::PatKind::TuplePat(p) => {
                let args = p.args().map(|p| self.collect_pat(p)).collect();
                Pat::Tuple(args)
            }
            ast::PatKind::PlaceholderPat(_) => Pat::Wild,
            ast::PatKind::StructPat(p) => {
                let path = p.path().and_then(Path::from_ast);
                let field_pat_list =
                    p.field_pat_list().expect("every struct should have a field list");
                let mut fields: Vec<_> = field_pat_list
                    .bind_pats()
                    .filter_map(|bind_pat| {
                        let ast_pat =
                            ast::Pat::cast(bind_pat.syntax().clone()).expect("bind pat is a pat");
                        let pat = self.collect_pat(ast_pat);
                        let name = bind_pat.name()?.as_name();
                        Some(FieldPat { name, pat })
                    })
                    .collect();
                let iter = field_pat_list.field_pats().filter_map(|f| {
                    let ast_pat = f.pat()?;
                    let pat = self.collect_pat(ast_pat);
                    let name = f.name()?.as_name();
                    Some(FieldPat { name, pat })
                });
                fields.extend(iter);

                Pat::Struct { path, args: fields }
            }

            // FIXME: implement
            ast::PatKind::LiteralPat(_) => Pat::Missing,
            ast::PatKind::SlicePat(_) | ast::PatKind::RangePat(_) => Pat::Missing,
        };
        let ptr = AstPtr::new(&pat);
        self.alloc_pat(pattern, Either::A(ptr))
    }

    fn collect_pat_opt(&mut self, pat: Option<ast::Pat>) -> PatId {
        if let Some(pat) = pat {
            self.collect_pat(pat)
        } else {
            self.pats.alloc(Pat::Missing)
        }
    }

    fn collect_const_body(&mut self, node: ast::ConstDef) {
        let body = self.collect_expr_opt(node.body());
        self.body_expr = Some(body);
    }

    fn collect_static_body(&mut self, node: ast::StaticDef) {
        let body = self.collect_expr_opt(node.body());
        self.body_expr = Some(body);
    }

    fn collect_fn_body(&mut self, node: ast::FnDef) {
        if let Some(param_list) = node.param_list() {
            if let Some(self_param) = param_list.self_param() {
                let ptr = AstPtr::new(&self_param);
                let param_pat = self.alloc_pat(
                    Pat::Bind {
                        name: SELF_PARAM,
                        mode: BindingAnnotation::Unannotated,
                        subpat: None,
                    },
                    Either::B(ptr),
                );
                self.params.push(param_pat);
            }

            for param in param_list.params() {
                let pat = if let Some(pat) = param.pat() {
                    pat
                } else {
                    continue;
                };
                let param_pat = self.collect_pat(pat);
                self.params.push(param_pat);
            }
        };

        let body = self.collect_block_opt(node.body());
        self.body_expr = Some(body);
    }

    fn finish(self) -> (Body, BodySourceMap) {
        let body = Body {
            owner: self.owner,
            exprs: self.exprs,
            pats: self.pats,
            params: self.params,
            body_expr: self.body_expr.expect("A body should have been collected"),
        };
        (body, self.source_map)
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

pub(crate) fn body_with_source_map_query(
    db: &impl HirDatabase,
    def: DefWithBody,
) -> (Arc<Body>, Arc<BodySourceMap>) {
    let mut collector;

    match def {
        DefWithBody::Const(ref c) => {
            let src = c.source(db);
            collector = ExprCollector::new(def, src.file_id, def.resolver(db), db);
            collector.collect_const_body(src.ast)
        }
        DefWithBody::Function(ref f) => {
            let src = f.source(db);
            collector = ExprCollector::new(def, src.file_id, def.resolver(db), db);
            collector.collect_fn_body(src.ast)
        }
        DefWithBody::Static(ref s) => {
            let src = s.source(db);
            collector = ExprCollector::new(def, src.file_id, def.resolver(db), db);
            collector.collect_static_body(src.ast)
        }
    }

    let (body, source_map) = collector.finish();
    (Arc::new(body), Arc::new(source_map))
}

pub(crate) fn body_hir_query(db: &impl HirDatabase, def: DefWithBody) -> Arc<Body> {
    db.body_with_source_map(def).0
}
