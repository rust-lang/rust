use std::ops::Index;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use ra_arena::{Arena, RawId, impl_arena_id, map::ArenaMap};
use ra_syntax::{
    SyntaxNodePtr, AstNode,
    ast::{self, LoopBodyOwner, ArgListOwner, NameOwner, LiteralFlavor, TypeAscriptionOwner}
};

use crate::{
    Path, Name, HirDatabase, Function, Resolver,
    name::AsName,
    type_ref::{Mutability, TypeRef},
};
use crate::{ path::GenericArgs, ty::primitive::{UintTy, UncertainIntTy, UncertainFloatTy}};

pub use self::scope::{ExprScopes, ScopesWithSourceMap, ScopeEntryWithSyntax};

pub(crate) mod scope;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprId(RawId);
impl_arena_id!(ExprId);

/// The body of an item (function, const etc.).
#[derive(Debug, Eq, PartialEq)]
pub struct Body {
    // TODO: this should be more general, consts & statics also have bodies
    /// The Function of the item this body belongs to
    owner: Function,
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
    // body: Arc<Body>,
    expr_map: FxHashMap<SyntaxNodePtr, ExprId>,
    expr_map_back: ArenaMap<ExprId, SyntaxNodePtr>,
    pat_map: FxHashMap<SyntaxNodePtr, PatId>,
    pat_map_back: ArenaMap<PatId, SyntaxNodePtr>,
}

impl Body {
    pub fn params(&self) -> &[PatId] {
        &self.params
    }

    pub fn body_expr(&self) -> ExprId {
        self.body_expr
    }

    pub fn owner(&self) -> Function {
        self.owner
    }

    pub fn exprs(&self) -> impl Iterator<Item = (ExprId, &Expr)> {
        self.exprs.iter()
    }

    pub fn pats(&self) -> impl Iterator<Item = (PatId, &Pat)> {
        self.pats.iter()
    }

    pub fn syntax_mapping(&self, db: &impl HirDatabase) -> Arc<BodySourceMap> {
        db.body_with_source_map(self.owner).1
    }
}

// needs arbitrary_self_types to be a method... or maybe move to the def?
pub fn resolver_for_expr(body: Arc<Body>, db: &impl HirDatabase, expr_id: ExprId) -> Resolver {
    let scopes = db.expr_scopes(body.owner);
    resolver_for_scope(body, db, scopes.scope_for(expr_id))
}

pub fn resolver_for_scope(
    body: Arc<Body>,
    db: &impl HirDatabase,
    scope_id: Option<scope::ScopeId>,
) -> Resolver {
    let mut r = body.owner.resolver(db);
    let scopes = db.expr_scopes(body.owner);
    let scope_chain = scopes.scope_chain_for(scope_id).collect::<Vec<_>>();
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
    pub fn expr_syntax(&self, expr: ExprId) -> Option<SyntaxNodePtr> {
        self.expr_map_back.get(expr).cloned()
    }

    pub fn syntax_expr(&self, ptr: SyntaxNodePtr) -> Option<ExprId> {
        self.expr_map.get(&ptr).cloned()
    }

    pub fn node_expr(&self, node: &ast::Expr) -> Option<ExprId> {
        self.expr_map.get(&SyntaxNodePtr::new(node.syntax())).cloned()
    }

    pub fn pat_syntax(&self, pat: PatId) -> Option<SyntaxNodePtr> {
        self.pat_map_back.get(pat).cloned()
    }

    pub fn syntax_pat(&self, ptr: SyntaxNodePtr) -> Option<PatId> {
        self.pat_map.get(&ptr).cloned()
    }

    pub fn node_pat(&self, node: &ast::Pat) -> Option<PatId> {
        self.pat_map.get(&SyntaxNodePtr::new(node.syntax())).cloned()
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
    Try {
        expr: ExprId,
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
    Lambda {
        args: Vec<PatId>,
        arg_types: Vec<Option<TypeRef>>,
        body: ExprId,
    },
    Tuple {
        exprs: Vec<ExprId>,
    },
    Array {
        exprs: Vec<ExprId>,
    },
    Literal(Literal),
}

pub use ra_syntax::ast::PrefixOp as UnaryOp;
pub use ra_syntax::ast::BinOp as BinaryOp;

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
            Expr::Field { expr, .. }
            | Expr::Try { expr }
            | Expr::Cast { expr, .. }
            | Expr::Ref { expr, .. }
            | Expr::UnaryOp { expr, .. } => {
                f(*expr);
            }
            Expr::Tuple { exprs } | Expr::Array { exprs } => {
                for expr in exprs {
                    f(*expr);
                }
            }
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
        // TODO: 'ellipsis' option
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
                subpat.iter().map(|pat| *pat).for_each(f);
            }
            Pat::Tuple(args) | Pat::TupleStruct { args, .. } => {
                args.iter().map(|pat| *pat).for_each(f);
            }
            Pat::Ref { pat, .. } => f(*pat),
            Pat::Slice { prefix, rest, suffix } => {
                let total_iter = prefix.iter().chain(rest.iter()).chain(suffix.iter());
                total_iter.map(|pat| *pat).for_each(f);
            }
            Pat::Struct { args, .. } => {
                args.iter().map(|f| f.pat).for_each(f);
            }
        }
    }
}

// Queries

struct ExprCollector {
    owner: Function,
    exprs: Arena<ExprId, Expr>,
    pats: Arena<PatId, Pat>,
    source_map: BodySourceMap,
    params: Vec<PatId>,
    body_expr: Option<ExprId>,
}

impl ExprCollector {
    fn new(owner: Function) -> Self {
        ExprCollector {
            owner,
            exprs: Arena::default(),
            pats: Arena::default(),
            source_map: BodySourceMap::default(),
            params: Vec::new(),
            body_expr: None,
        }
    }

    fn alloc_expr(&mut self, expr: Expr, syntax_ptr: SyntaxNodePtr) -> ExprId {
        let id = self.exprs.alloc(expr);
        self.source_map.expr_map.insert(syntax_ptr, id);
        self.source_map.expr_map_back.insert(id, syntax_ptr);
        id
    }

    fn alloc_pat(&mut self, pat: Pat, syntax_ptr: SyntaxNodePtr) -> PatId {
        let id = self.pats.alloc(pat);
        self.source_map.pat_map.insert(syntax_ptr, id);
        self.source_map.pat_map_back.insert(id, syntax_ptr);
        id
    }

    fn empty_block(&mut self) -> ExprId {
        let block = Expr::Block { statements: Vec::new(), tail: None };
        self.exprs.alloc(block)
    }

    fn collect_expr(&mut self, expr: &ast::Expr) -> ExprId {
        let syntax_ptr = SyntaxNodePtr::new(expr.syntax());
        match expr.kind() {
            ast::ExprKind::IfExpr(e) => {
                if let Some(pat) = e.condition().and_then(|c| c.pat()) {
                    // if let -- desugar to match
                    let pat = self.collect_pat(pat);
                    let match_expr =
                        self.collect_expr_opt(e.condition().expect("checked above").expr());
                    let then_branch = self.collect_block_opt(e.then_branch());
                    let else_branch = e
                        .else_branch()
                        .map(|b| match b {
                            ast::ElseBranchFlavor::Block(it) => self.collect_block(it),
                            ast::ElseBranchFlavor::IfExpr(elif) => {
                                let expr: &ast::Expr = ast::Expr::cast(elif.syntax()).unwrap();
                                self.collect_expr(expr)
                            }
                        })
                        .unwrap_or_else(|| self.empty_block());
                    let placeholder_pat = self.pats.alloc(Pat::Missing);
                    let arms = vec![
                        MatchArm { pats: vec![pat], expr: then_branch, guard: None },
                        MatchArm { pats: vec![placeholder_pat], expr: else_branch, guard: None },
                    ];
                    self.alloc_expr(Expr::Match { expr: match_expr, arms }, syntax_ptr)
                } else {
                    let condition = self.collect_expr_opt(e.condition().and_then(|c| c.expr()));
                    let then_branch = self.collect_block_opt(e.then_branch());
                    let else_branch = e.else_branch().map(|b| match b {
                        ast::ElseBranchFlavor::Block(it) => self.collect_block(it),
                        ast::ElseBranchFlavor::IfExpr(elif) => {
                            let expr: &ast::Expr = ast::Expr::cast(elif.syntax()).unwrap();
                            self.collect_expr(expr)
                        }
                    });
                    self.alloc_expr(Expr::If { condition, then_branch, else_branch }, syntax_ptr)
                }
            }
            ast::ExprKind::BlockExpr(e) => self.collect_block_opt(e.block()),
            ast::ExprKind::LoopExpr(e) => {
                let body = self.collect_block_opt(e.loop_body());
                self.alloc_expr(Expr::Loop { body }, syntax_ptr)
            }
            ast::ExprKind::WhileExpr(e) => {
                let condition = if let Some(condition) = e.condition() {
                    if condition.pat().is_none() {
                        self.collect_expr_opt(condition.expr())
                    } else {
                        // TODO handle while let
                        return self.alloc_expr(Expr::Missing, syntax_ptr);
                    }
                } else {
                    self.exprs.alloc(Expr::Missing)
                };
                let body = self.collect_block_opt(e.loop_body());
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
                // TODO: labels
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
                let fields = if let Some(nfl) = e.named_field_list() {
                    nfl.fields()
                        .map(|field| StructLitField {
                            name: field
                                .name_ref()
                                .map(|nr| nr.as_name())
                                .unwrap_or_else(Name::missing),
                            expr: if let Some(e) = field.expr() {
                                self.collect_expr(e)
                            } else if let Some(nr) = field.name_ref() {
                                // field shorthand
                                let id = self.exprs.alloc(Expr::Path(Path::from_name_ref(nr)));
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
                        .collect()
                } else {
                    Vec::new()
                };
                let spread = e.spread().map(|s| self.collect_expr(s));
                self.alloc_expr(Expr::StructLit { path, fields, spread }, syntax_ptr)
            }
            ast::ExprKind::FieldExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let name = e.name_ref().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                self.alloc_expr(Expr::Field { expr, name }, syntax_ptr)
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
                if let Some(op) = e.op() {
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
                let op = e.op();
                self.alloc_expr(Expr::BinaryOp { lhs, rhs, op }, syntax_ptr)
            }
            ast::ExprKind::TupleExpr(e) => {
                let exprs = e.exprs().map(|expr| self.collect_expr(expr)).collect();
                self.alloc_expr(Expr::Tuple { exprs }, syntax_ptr)
            }
            ast::ExprKind::ArrayExpr(e) => {
                let exprs = e.exprs().map(|expr| self.collect_expr(expr)).collect();
                self.alloc_expr(Expr::Array { exprs }, syntax_ptr)
            }
            ast::ExprKind::Literal(e) => {
                let child = if let Some(child) = e.literal_expr() {
                    child
                } else {
                    return self.alloc_expr(Expr::Missing, syntax_ptr);
                };

                let lit = match child.flavor() {
                    LiteralFlavor::IntNumber { suffix } => {
                        let known_name =
                            suffix.map(Name::new).and_then(|name| UncertainIntTy::from_name(&name));

                        Literal::Int(
                            Default::default(),
                            known_name.unwrap_or(UncertainIntTy::Unknown),
                        )
                    }
                    LiteralFlavor::FloatNumber { suffix } => {
                        let known_name = suffix
                            .map(Name::new)
                            .and_then(|name| UncertainFloatTy::from_name(&name));

                        Literal::Float(
                            Default::default(),
                            known_name.unwrap_or(UncertainFloatTy::Unknown),
                        )
                    }
                    LiteralFlavor::ByteString => Literal::ByteString(Default::default()),
                    LiteralFlavor::String => Literal::String(Default::default()),
                    LiteralFlavor::Byte => {
                        Literal::Int(Default::default(), UncertainIntTy::Unsigned(UintTy::U8))
                    }
                    LiteralFlavor::Bool => Literal::Bool(Default::default()),
                    LiteralFlavor::Char => Literal::Char(Default::default()),
                };
                self.alloc_expr(Expr::Literal(lit), syntax_ptr)
            }

            // TODO implement HIR for these:
            ast::ExprKind::Label(_e) => self.alloc_expr(Expr::Missing, syntax_ptr),
            ast::ExprKind::IndexExpr(_e) => self.alloc_expr(Expr::Missing, syntax_ptr),
            ast::ExprKind::RangeExpr(_e) => self.alloc_expr(Expr::Missing, syntax_ptr),
        }
    }

    fn collect_expr_opt(&mut self, expr: Option<&ast::Expr>) -> ExprId {
        if let Some(expr) = expr {
            self.collect_expr(expr)
        } else {
            self.exprs.alloc(Expr::Missing)
        }
    }

    fn collect_block(&mut self, block: &ast::Block) -> ExprId {
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

    fn collect_block_opt(&mut self, block: Option<&ast::Block>) -> ExprId {
        if let Some(block) = block {
            self.collect_block(block)
        } else {
            self.exprs.alloc(Expr::Missing)
        }
    }

    fn collect_pat(&mut self, pat: &ast::Pat) -> PatId {
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
                        let ast_pat = ast::Pat::cast(bind_pat.syntax()).expect("bind pat is a pat");
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

            // TODO: implement
            ast::PatKind::LiteralPat(_) => Pat::Missing,
            ast::PatKind::SlicePat(_) | ast::PatKind::RangePat(_) => Pat::Missing,
        };
        let syntax_ptr = SyntaxNodePtr::new(pat.syntax());
        self.alloc_pat(pattern, syntax_ptr)
    }

    fn collect_pat_opt(&mut self, pat: Option<&ast::Pat>) -> PatId {
        if let Some(pat) = pat {
            self.collect_pat(pat)
        } else {
            self.pats.alloc(Pat::Missing)
        }
    }

    fn collect_fn_body(&mut self, node: &ast::FnDef) {
        if let Some(param_list) = node.param_list() {
            if let Some(self_param) = param_list.self_param() {
                let self_param = SyntaxNodePtr::new(
                    self_param.self_kw().expect("self param without self keyword").syntax(),
                );
                let param_pat = self.alloc_pat(
                    Pat::Bind {
                        name: Name::self_param(),
                        mode: BindingAnnotation::Unannotated,
                        subpat: None,
                    },
                    self_param,
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

pub(crate) fn body_with_source_map_query(
    db: &impl HirDatabase,
    func: Function,
) -> (Arc<Body>, Arc<BodySourceMap>) {
    let mut collector = ExprCollector::new(func);

    // TODO: consts, etc.
    collector.collect_fn_body(&func.source(db).1);

    let (body, source_map) = collector.finish();
    (Arc::new(body), Arc::new(source_map))
}

pub(crate) fn body_hir_query(db: &impl HirDatabase, func: Function) -> Arc<Body> {
    db.body_with_source_map(func).0
}

#[cfg(test)]
fn collect_fn_body_syntax(function: Function, node: &ast::FnDef) -> (Body, BodySourceMap) {
    let mut collector = ExprCollector::new(function);
    collector.collect_fn_body(node);
    collector.finish()
}
