//! Defines `Body`: a lowered representation of bodies of functions, statics and
//! consts.
mod lower;
#[cfg(test)]
mod tests;
pub mod scope;
mod pretty;

use std::ops::Index;

use base_db::CrateId;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{name::Name, HirFileId, InFile};
use la_arena::{Arena, ArenaMap};
use profile::Count;
use rustc_hash::FxHashMap;
use syntax::{ast, AstPtr, SyntaxNodePtr};
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    expander::Expander,
    hir::{
        dummy_expr_id, Binding, BindingId, Expr, ExprId, Label, LabelId, Pat, PatId, RecordFieldPat,
    },
    nameres::DefMap,
    path::{ModPath, Path},
    src::{HasChildSource, HasSource},
    BlockId, DefWithBodyId, HasModule, Lookup,
};

/// The body of an item (function, const etc.).
#[derive(Debug, Eq, PartialEq)]
pub struct Body {
    pub exprs: Arena<Expr>,
    pub pats: Arena<Pat>,
    pub bindings: Arena<Binding>,
    pub labels: Arena<Label>,
    /// Id of the closure/generator that owns the corresponding binding. If a binding is owned by the
    /// top level expression, it will not be listed in here.
    pub binding_owners: FxHashMap<BindingId, ExprId>,
    /// The patterns for the function's parameters. While the parameter types are
    /// part of the function signature, the patterns are not (they don't change
    /// the external type of the function).
    ///
    /// If this `Body` is for the body of a constant, this will just be
    /// empty.
    pub params: Vec<PatId>,
    /// The `ExprId` of the actual body expression.
    pub body_expr: ExprId,
    /// Block expressions in this body that may contain inner items.
    block_scopes: Vec<BlockId>,
    _c: Count<Self>,
}

pub type ExprPtr = AstPtr<ast::Expr>;
pub type ExprSource = InFile<ExprPtr>;

pub type PatPtr = Either<AstPtr<ast::Pat>, AstPtr<ast::SelfParam>>;
pub type PatSource = InFile<PatPtr>;

pub type LabelPtr = AstPtr<ast::Label>;
pub type LabelSource = InFile<LabelPtr>;

pub type FieldPtr = AstPtr<ast::RecordExprField>;
pub type FieldSource = InFile<FieldPtr>;

/// An item body together with the mapping from syntax nodes to HIR expression
/// IDs. This is needed to go from e.g. a position in a file to the HIR
/// expression containing it; but for type inference etc., we want to operate on
/// a structure that is agnostic to the actual positions of expressions in the
/// file, so that we don't recompute types whenever some whitespace is typed.
///
/// One complication here is that, due to macro expansion, a single `Body` might
/// be spread across several files. So, for each ExprId and PatId, we record
/// both the HirFileId and the position inside the file. However, we only store
/// AST -> ExprId mapping for non-macro files, as it is not clear how to handle
/// this properly for macros.
#[derive(Default, Debug, Eq, PartialEq)]
pub struct BodySourceMap {
    expr_map: FxHashMap<ExprSource, ExprId>,
    expr_map_back: ArenaMap<ExprId, ExprSource>,

    pat_map: FxHashMap<PatSource, PatId>,
    pat_map_back: ArenaMap<PatId, PatSource>,

    label_map: FxHashMap<LabelSource, LabelId>,
    label_map_back: ArenaMap<LabelId, LabelSource>,

    /// We don't create explicit nodes for record fields (`S { record_field: 92 }`).
    /// Instead, we use id of expression (`92`) to identify the field.
    field_map: FxHashMap<FieldSource, ExprId>,
    field_map_back: FxHashMap<ExprId, FieldSource>,

    expansions: FxHashMap<InFile<AstPtr<ast::MacroCall>>, HirFileId>,

    /// Diagnostics accumulated during body lowering. These contain `AstPtr`s and so are stored in
    /// the source map (since they're just as volatile).
    diagnostics: Vec<BodyDiagnostic>,
}

#[derive(Default, Debug, Eq, PartialEq, Clone, Copy)]
pub struct SyntheticSyntax;

#[derive(Debug, Eq, PartialEq)]
pub enum BodyDiagnostic {
    InactiveCode { node: InFile<SyntaxNodePtr>, cfg: CfgExpr, opts: CfgOptions },
    MacroError { node: InFile<AstPtr<ast::MacroCall>>, message: String },
    UnresolvedProcMacro { node: InFile<AstPtr<ast::MacroCall>>, krate: CrateId },
    UnresolvedMacroCall { node: InFile<AstPtr<ast::MacroCall>>, path: ModPath },
    UnreachableLabel { node: InFile<AstPtr<ast::Lifetime>>, name: Name },
    UndeclaredLabel { node: InFile<AstPtr<ast::Lifetime>>, name: Name },
}

impl Body {
    pub(crate) fn body_with_source_map_query(
        db: &dyn DefDatabase,
        def: DefWithBodyId,
    ) -> (Arc<Body>, Arc<BodySourceMap>) {
        let _p = profile::span("body_with_source_map_query");
        let mut params = None;

        let mut is_async_fn = false;
        let InFile { file_id, value: body } = {
            match def {
                DefWithBodyId::FunctionId(f) => {
                    let data = db.function_data(f);
                    let f = f.lookup(db);
                    let src = f.source(db);
                    params = src.value.param_list().map(|param_list| {
                        let item_tree = f.id.item_tree(db);
                        let func = &item_tree[f.id.value];
                        let krate = f.container.module(db).krate;
                        let crate_graph = db.crate_graph();
                        (
                            param_list,
                            func.params.clone().map(move |param| {
                                item_tree
                                    .attrs(db, krate, param.into())
                                    .is_cfg_enabled(&crate_graph[krate].cfg_options)
                            }),
                        )
                    });
                    is_async_fn = data.has_async_kw();
                    src.map(|it| it.body().map(ast::Expr::from))
                }
                DefWithBodyId::ConstId(c) => {
                    let c = c.lookup(db);
                    let src = c.source(db);
                    src.map(|it| it.body())
                }
                DefWithBodyId::StaticId(s) => {
                    let s = s.lookup(db);
                    let src = s.source(db);
                    src.map(|it| it.body())
                }
                DefWithBodyId::VariantId(v) => {
                    let src = v.parent.child_source(db);
                    src.map(|it| it[v.local_id].expr())
                }
                DefWithBodyId::InTypeConstId(c) => c.lookup(db).id.map(|_| c.source(db).expr()),
            }
        };
        let module = def.module(db);
        let expander = Expander::new(db, file_id, module);
        let (mut body, source_map) =
            Body::new(db, def, expander, params, body, module.krate, is_async_fn);
        body.shrink_to_fit();

        (Arc::new(body), Arc::new(source_map))
    }

    pub(crate) fn body_query(db: &dyn DefDatabase, def: DefWithBodyId) -> Arc<Body> {
        db.body_with_source_map(def).0
    }

    /// Returns an iterator over all block expressions in this body that define inner items.
    pub fn blocks<'a>(
        &'a self,
        db: &'a dyn DefDatabase,
    ) -> impl Iterator<Item = (BlockId, Arc<DefMap>)> + '_ {
        self.block_scopes.iter().map(move |&block| (block, db.block_def_map(block)))
    }

    pub fn pretty_print(&self, db: &dyn DefDatabase, owner: DefWithBodyId) -> String {
        pretty::print_body_hir(db, self, owner)
    }

    pub fn pretty_print_expr(
        &self,
        db: &dyn DefDatabase,
        owner: DefWithBodyId,
        expr: ExprId,
    ) -> String {
        pretty::print_expr_hir(db, self, owner, expr)
    }

    fn new(
        db: &dyn DefDatabase,
        owner: DefWithBodyId,
        expander: Expander,
        params: Option<(ast::ParamList, impl Iterator<Item = bool>)>,
        body: Option<ast::Expr>,
        krate: CrateId,
        is_async_fn: bool,
    ) -> (Body, BodySourceMap) {
        lower::lower(db, owner, expander, params, body, krate, is_async_fn)
    }

    fn shrink_to_fit(&mut self) {
        let Self {
            _c: _,
            body_expr: _,
            block_scopes,
            exprs,
            labels,
            params,
            pats,
            bindings,
            binding_owners,
        } = self;
        block_scopes.shrink_to_fit();
        exprs.shrink_to_fit();
        labels.shrink_to_fit();
        params.shrink_to_fit();
        pats.shrink_to_fit();
        bindings.shrink_to_fit();
        binding_owners.shrink_to_fit();
    }

    pub fn walk_bindings_in_pat(&self, pat_id: PatId, mut f: impl FnMut(BindingId)) {
        self.walk_pats(pat_id, &mut |pat| {
            if let Pat::Bind { id, .. } = &self[pat] {
                f(*id);
            }
        });
    }

    pub fn walk_pats_shallow(&self, pat_id: PatId, mut f: impl FnMut(PatId)) {
        let pat = &self[pat_id];
        match pat {
            Pat::Range { .. }
            | Pat::Lit(..)
            | Pat::Path(..)
            | Pat::ConstBlock(..)
            | Pat::Wild
            | Pat::Missing => {}
            &Pat::Bind { subpat, .. } => {
                if let Some(subpat) = subpat {
                    f(subpat);
                }
            }
            Pat::Or(args) | Pat::Tuple { args, .. } | Pat::TupleStruct { args, .. } => {
                args.iter().copied().for_each(|p| f(p));
            }
            Pat::Ref { pat, .. } => f(*pat),
            Pat::Slice { prefix, slice, suffix } => {
                let total_iter = prefix.iter().chain(slice.iter()).chain(suffix.iter());
                total_iter.copied().for_each(|p| f(p));
            }
            Pat::Record { args, .. } => {
                args.iter().for_each(|RecordFieldPat { pat, .. }| f(*pat));
            }
            Pat::Box { inner } => f(*inner),
        }
    }

    pub fn walk_pats(&self, pat_id: PatId, f: &mut impl FnMut(PatId)) {
        f(pat_id);
        self.walk_pats_shallow(pat_id, |p| self.walk_pats(p, f));
    }

    pub fn is_binding_upvar(&self, binding: BindingId, relative_to: ExprId) -> bool {
        match self.binding_owners.get(&binding) {
            Some(it) => {
                // We assign expression ids in a way that outer closures will receive
                // a lower id
                it.into_raw() < relative_to.into_raw()
            }
            None => true,
        }
    }
}

impl Default for Body {
    fn default() -> Self {
        Self {
            body_expr: dummy_expr_id(),
            exprs: Default::default(),
            pats: Default::default(),
            bindings: Default::default(),
            labels: Default::default(),
            params: Default::default(),
            block_scopes: Default::default(),
            binding_owners: Default::default(),
            _c: Default::default(),
        }
    }
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

impl Index<LabelId> for Body {
    type Output = Label;

    fn index(&self, label: LabelId) -> &Label {
        &self.labels[label]
    }
}

impl Index<BindingId> for Body {
    type Output = Binding;

    fn index(&self, b: BindingId) -> &Binding {
        &self.bindings[b]
    }
}

// FIXME: Change `node_` prefix to something more reasonable.
// Perhaps `expr_syntax` and `expr_id`?
impl BodySourceMap {
    pub fn expr_syntax(&self, expr: ExprId) -> Result<ExprSource, SyntheticSyntax> {
        self.expr_map_back.get(expr).cloned().ok_or(SyntheticSyntax)
    }

    pub fn node_expr(&self, node: InFile<&ast::Expr>) -> Option<ExprId> {
        let src = node.map(AstPtr::new);
        self.expr_map.get(&src).cloned()
    }

    pub fn node_macro_file(&self, node: InFile<&ast::MacroCall>) -> Option<HirFileId> {
        let src = node.map(AstPtr::new);
        self.expansions.get(&src).cloned()
    }

    pub fn pat_syntax(&self, pat: PatId) -> Result<PatSource, SyntheticSyntax> {
        self.pat_map_back.get(pat).cloned().ok_or(SyntheticSyntax)
    }

    pub fn node_pat(&self, node: InFile<&ast::Pat>) -> Option<PatId> {
        let src = node.map(|it| Either::Left(AstPtr::new(it)));
        self.pat_map.get(&src).cloned()
    }

    pub fn node_self_param(&self, node: InFile<&ast::SelfParam>) -> Option<PatId> {
        let src = node.map(|it| Either::Right(AstPtr::new(it)));
        self.pat_map.get(&src).cloned()
    }

    pub fn label_syntax(&self, label: LabelId) -> LabelSource {
        self.label_map_back[label].clone()
    }

    pub fn node_label(&self, node: InFile<&ast::Label>) -> Option<LabelId> {
        let src = node.map(AstPtr::new);
        self.label_map.get(&src).cloned()
    }

    pub fn field_syntax(&self, expr: ExprId) -> FieldSource {
        self.field_map_back[&expr].clone()
    }

    pub fn node_field(&self, node: InFile<&ast::RecordExprField>) -> Option<ExprId> {
        let src = node.map(AstPtr::new);
        self.field_map.get(&src).cloned()
    }

    pub fn macro_expansion_expr(&self, node: InFile<&ast::MacroExpr>) -> Option<ExprId> {
        let src = node.map(AstPtr::new).map(AstPtr::upcast::<ast::MacroExpr>).map(AstPtr::upcast);
        self.expr_map.get(&src).copied()
    }

    /// Get a reference to the body source map's diagnostics.
    pub fn diagnostics(&self) -> &[BodyDiagnostic] {
        &self.diagnostics
    }
}
