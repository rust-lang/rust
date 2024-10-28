//! Defines `Body`: a lowered representation of bodies of functions, statics and
//! consts.
mod lower;
mod pretty;
pub mod scope;
#[cfg(test)]
mod tests;

use std::ops::{Deref, Index};

use base_db::CrateId;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{name::Name, ExpandError, InFile};
use la_arena::{Arena, ArenaMap, Idx, RawIdx};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use span::{Edition, MacroFileId};
use syntax::{ast, AstPtr, SyntaxNodePtr};
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    expander::Expander,
    hir::{
        dummy_expr_id, Array, AsmOperand, Binding, BindingId, Expr, ExprId, ExprOrPatId, Label,
        LabelId, Pat, PatId, RecordFieldPat, Statement,
    },
    item_tree::AttrOwner,
    nameres::DefMap,
    path::{ModPath, Path},
    src::HasSource,
    type_ref::{TypeRef, TypeRefId, TypesMap, TypesSourceMap},
    BlockId, DefWithBodyId, HasModule, Lookup,
};

/// A wrapper around [`span::SyntaxContextId`] that is intended only for comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HygieneId(pub(crate) span::SyntaxContextId);

impl HygieneId {
    pub const ROOT: Self = Self(span::SyntaxContextId::ROOT);

    pub fn new(ctx: span::SyntaxContextId) -> Self {
        Self(ctx)
    }

    pub(crate) fn is_root(self) -> bool {
        self.0.is_root()
    }
}

/// The body of an item (function, const etc.).
#[derive(Debug, Eq, PartialEq)]
pub struct Body {
    pub exprs: Arena<Expr>,
    pub pats: Arena<Pat>,
    pub bindings: Arena<Binding>,
    pub labels: Arena<Label>,
    /// Id of the closure/coroutine that owns the corresponding binding. If a binding is owned by the
    /// top level expression, it will not be listed in here.
    pub binding_owners: FxHashMap<BindingId, ExprId>,
    /// The patterns for the function's parameters. While the parameter types are
    /// part of the function signature, the patterns are not (they don't change
    /// the external type of the function).
    ///
    /// If this `Body` is for the body of a constant, this will just be
    /// empty.
    pub params: Box<[PatId]>,
    pub self_param: Option<BindingId>,
    /// The `ExprId` of the actual body expression.
    pub body_expr: ExprId,
    pub types: TypesMap,
    /// Block expressions in this body that may contain inner items.
    block_scopes: Vec<BlockId>,

    /// A map from binding to its hygiene ID.
    ///
    /// Bindings that don't come from macro expansion are not allocated to save space, so not all bindings appear here.
    /// If a binding does not appear here it has `SyntaxContextId::ROOT`.
    ///
    /// Note that this may not be the direct `SyntaxContextId` of the binding's expansion, because transparent
    /// expansions are attributed to their parent expansion (recursively).
    binding_hygiene: FxHashMap<BindingId, HygieneId>,
    /// A map from an variable usages to their hygiene ID.
    ///
    /// Expressions that can be recorded here are single segment path, although not all single segments path refer
    /// to variables and have hygiene (some refer to items, we don't know at this stage).
    expr_hygiene: FxHashMap<ExprId, HygieneId>,
    /// A map from a destructuring assignment possible variable usages to their hygiene ID.
    pat_hygiene: FxHashMap<PatId, HygieneId>,
}

pub type ExprPtr = AstPtr<ast::Expr>;
pub type ExprSource = InFile<ExprPtr>;

pub type PatPtr = AstPtr<ast::Pat>;
pub type PatSource = InFile<PatPtr>;

pub type LabelPtr = AstPtr<ast::Label>;
pub type LabelSource = InFile<LabelPtr>;

pub type FieldPtr = AstPtr<ast::RecordExprField>;
pub type FieldSource = InFile<FieldPtr>;

pub type PatFieldPtr = AstPtr<Either<ast::RecordExprField, ast::RecordPatField>>;
pub type PatFieldSource = InFile<PatFieldPtr>;

pub type ExprOrPatPtr = AstPtr<Either<ast::Expr, ast::Pat>>;
pub type ExprOrPatSource = InFile<ExprOrPatPtr>;

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
    // AST expressions can create patterns in destructuring assignments. Therefore, `ExprSource` can also map
    // to `PatId`, and `PatId` can also map to `ExprSource` (the other way around is unaffected).
    expr_map: FxHashMap<ExprSource, ExprOrPatId>,
    expr_map_back: ArenaMap<ExprId, ExprSource>,

    pat_map: FxHashMap<PatSource, PatId>,
    pat_map_back: ArenaMap<PatId, ExprOrPatSource>,

    label_map: FxHashMap<LabelSource, LabelId>,
    label_map_back: ArenaMap<LabelId, LabelSource>,

    self_param: Option<InFile<AstPtr<ast::SelfParam>>>,
    binding_definitions: FxHashMap<BindingId, SmallVec<[PatId; 4]>>,

    /// We don't create explicit nodes for record fields (`S { record_field: 92 }`).
    /// Instead, we use id of expression (`92`) to identify the field.
    field_map_back: FxHashMap<ExprId, FieldSource>,
    pat_field_map_back: FxHashMap<PatId, PatFieldSource>,

    types: TypesSourceMap,

    // FIXME: Make this a sane struct.
    template_map: Option<
        Box<(
            // format_args!
            FxHashMap<ExprId, (HygieneId, Vec<(syntax::TextRange, Name)>)>,
            // asm!
            FxHashMap<ExprId, Vec<Vec<(syntax::TextRange, usize)>>>,
        )>,
    >,

    expansions: FxHashMap<InFile<AstPtr<ast::MacroCall>>, MacroFileId>,

    /// Diagnostics accumulated during body lowering. These contain `AstPtr`s and so are stored in
    /// the source map (since they're just as volatile).
    diagnostics: Vec<BodyDiagnostic>,
}

#[derive(Default, Debug, Eq, PartialEq, Clone, Copy)]
pub struct SyntheticSyntax;

#[derive(Debug, Eq, PartialEq)]
pub enum BodyDiagnostic {
    InactiveCode { node: InFile<SyntaxNodePtr>, cfg: CfgExpr, opts: CfgOptions },
    MacroError { node: InFile<AstPtr<ast::MacroCall>>, err: ExpandError },
    UnresolvedMacroCall { node: InFile<AstPtr<ast::MacroCall>>, path: ModPath },
    UnreachableLabel { node: InFile<AstPtr<ast::Lifetime>>, name: Name },
    AwaitOutsideOfAsync { node: InFile<AstPtr<ast::AwaitExpr>>, location: String },
    UndeclaredLabel { node: InFile<AstPtr<ast::Lifetime>>, name: Name },
}

impl Body {
    pub(crate) fn body_with_source_map_query(
        db: &dyn DefDatabase,
        def: DefWithBodyId,
    ) -> (Arc<Body>, Arc<BodySourceMap>) {
        let _p = tracing::info_span!("body_with_source_map_query").entered();
        let mut params = None;

        let mut is_async_fn = false;
        let InFile { file_id, value: body } = {
            match def {
                DefWithBodyId::FunctionId(f) => {
                    let data = db.function_data(f);
                    let f = f.lookup(db);
                    let src = f.source(db);
                    params = src.value.param_list().map(move |param_list| {
                        let item_tree = f.id.item_tree(db);
                        let func = &item_tree[f.id.value];
                        let krate = f.container.module(db).krate;
                        let crate_graph = db.crate_graph();
                        (
                            param_list,
                            (0..func.params.len()).map(move |idx| {
                                item_tree
                                    .attrs(
                                        db,
                                        krate,
                                        AttrOwner::Param(
                                            f.id.value,
                                            Idx::from_raw(RawIdx::from(idx as u32)),
                                        ),
                                    )
                                    .is_cfg_enabled(&crate_graph[krate].cfg_options)
                            }),
                        )
                    });
                    is_async_fn = data.is_async();
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
                    let s = v.lookup(db);
                    let src = s.source(db);
                    src.map(|it| it.expr())
                }
                DefWithBodyId::InTypeConstId(c) => c.lookup(db).id.map(|_| c.source(db).expr()),
            }
        };
        let module = def.module(db);
        let expander = Expander::new(db, file_id, module);
        let (mut body, mut source_map) =
            Body::new(db, def, expander, params, body, module.krate, is_async_fn);
        body.shrink_to_fit();
        source_map.shrink_to_fit();

        (Arc::new(body), Arc::new(source_map))
    }

    pub(crate) fn body_query(db: &dyn DefDatabase, def: DefWithBodyId) -> Arc<Body> {
        db.body_with_source_map(def).0
    }

    /// Returns an iterator over all block expressions in this body that define inner items.
    pub fn blocks<'a>(
        &'a self,
        db: &'a dyn DefDatabase,
    ) -> impl Iterator<Item = (BlockId, Arc<DefMap>)> + 'a {
        self.block_scopes.iter().map(move |&block| (block, db.block_def_map(block)))
    }

    pub fn pretty_print(
        &self,
        db: &dyn DefDatabase,
        owner: DefWithBodyId,
        edition: Edition,
    ) -> String {
        pretty::print_body_hir(db, self, owner, edition)
    }

    pub fn pretty_print_expr(
        &self,
        db: &dyn DefDatabase,
        owner: DefWithBodyId,
        expr: ExprId,
        edition: Edition,
    ) -> String {
        pretty::print_expr_hir(db, self, owner, expr, edition)
    }

    pub fn pretty_print_pat(
        &self,
        db: &dyn DefDatabase,
        owner: DefWithBodyId,
        pat: PatId,
        oneline: bool,
        edition: Edition,
    ) -> String {
        pretty::print_pat_hir(db, self, owner, pat, oneline, edition)
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
            body_expr: _,
            params: _,
            self_param: _,
            block_scopes,
            exprs,
            labels,
            pats,
            bindings,
            binding_owners,
            binding_hygiene,
            expr_hygiene,
            pat_hygiene,
            types,
        } = self;
        block_scopes.shrink_to_fit();
        exprs.shrink_to_fit();
        labels.shrink_to_fit();
        pats.shrink_to_fit();
        bindings.shrink_to_fit();
        binding_owners.shrink_to_fit();
        binding_hygiene.shrink_to_fit();
        expr_hygiene.shrink_to_fit();
        pat_hygiene.shrink_to_fit();
        types.shrink_to_fit();
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
            | Pat::Missing
            | Pat::Expr(_) => {}
            &Pat::Bind { subpat, .. } => {
                if let Some(subpat) = subpat {
                    f(subpat);
                }
            }
            Pat::Or(args) | Pat::Tuple { args, .. } | Pat::TupleStruct { args, .. } => {
                args.iter().copied().for_each(f);
            }
            Pat::Ref { pat, .. } => f(*pat),
            Pat::Slice { prefix, slice, suffix } => {
                let total_iter = prefix.iter().chain(slice.iter()).chain(suffix.iter());
                total_iter.copied().for_each(f);
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

    pub fn walk_child_exprs(&self, expr_id: ExprId, mut f: impl FnMut(ExprId)) {
        let expr = &self[expr_id];
        match expr {
            Expr::Continue { .. }
            | Expr::Const(_)
            | Expr::Missing
            | Expr::Path(_)
            | Expr::OffsetOf(_)
            | Expr::Literal(_)
            | Expr::Underscore => {}
            Expr::InlineAsm(it) => it.operands.iter().for_each(|(_, op)| match op {
                AsmOperand::In { expr, .. }
                | AsmOperand::Out { expr: Some(expr), .. }
                | AsmOperand::InOut { expr, .. } => f(*expr),
                AsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                    f(*in_expr);
                    if let Some(out_expr) = out_expr {
                        f(*out_expr);
                    }
                }
                AsmOperand::Out { expr: None, .. }
                | AsmOperand::Const(_)
                | AsmOperand::Label(_)
                | AsmOperand::Sym(_) => (),
            }),
            Expr::If { condition, then_branch, else_branch } => {
                f(*condition);
                f(*then_branch);
                if let &Some(else_branch) = else_branch {
                    f(else_branch);
                }
            }
            Expr::Let { expr, .. } => {
                f(*expr);
            }
            Expr::Block { statements, tail, .. }
            | Expr::Unsafe { statements, tail, .. }
            | Expr::Async { statements, tail, .. } => {
                for stmt in statements.iter() {
                    match stmt {
                        Statement::Let { initializer, else_branch, pat, .. } => {
                            if let &Some(expr) = initializer {
                                f(expr);
                            }
                            if let &Some(expr) = else_branch {
                                f(expr);
                            }
                            self.walk_exprs_in_pat(*pat, &mut f);
                        }
                        Statement::Expr { expr: expression, .. } => f(*expression),
                        Statement::Item(_) => (),
                    }
                }
                if let &Some(expr) = tail {
                    f(expr);
                }
            }
            Expr::Loop { body, .. } => f(*body),
            Expr::Call { callee, args, .. } => {
                f(*callee);
                args.iter().copied().for_each(f);
            }
            Expr::MethodCall { receiver, args, .. } => {
                f(*receiver);
                args.iter().copied().for_each(f);
            }
            Expr::Match { expr, arms } => {
                f(*expr);
                arms.iter().map(|arm| arm.expr).for_each(f);
            }
            Expr::Break { expr, .. }
            | Expr::Return { expr }
            | Expr::Yield { expr }
            | Expr::Yeet { expr } => {
                if let &Some(expr) = expr {
                    f(expr);
                }
            }
            Expr::Become { expr } => f(*expr),
            Expr::RecordLit { fields, spread, .. } => {
                for field in fields.iter() {
                    f(field.expr);
                }
                if let &Some(expr) = spread {
                    f(expr);
                }
            }
            Expr::Closure { body, .. } => {
                f(*body);
            }
            Expr::BinaryOp { lhs, rhs, .. } => {
                f(*lhs);
                f(*rhs);
            }
            Expr::Range { lhs, rhs, .. } => {
                if let &Some(lhs) = rhs {
                    f(lhs);
                }
                if let &Some(rhs) = lhs {
                    f(rhs);
                }
            }
            Expr::Index { base, index, .. } => {
                f(*base);
                f(*index);
            }
            Expr::Field { expr, .. }
            | Expr::Await { expr }
            | Expr::Cast { expr, .. }
            | Expr::Ref { expr, .. }
            | Expr::UnaryOp { expr, .. }
            | Expr::Box { expr } => {
                f(*expr);
            }
            Expr::Tuple { exprs, .. } => exprs.iter().copied().for_each(f),
            Expr::Array(a) => match a {
                Array::ElementList { elements, .. } => elements.iter().copied().for_each(f),
                Array::Repeat { initializer, repeat } => {
                    f(*initializer);
                    f(*repeat)
                }
            },
            &Expr::Assignment { target, value } => {
                self.walk_exprs_in_pat(target, &mut f);
                f(value);
            }
        }
    }

    pub fn walk_exprs_in_pat(&self, pat_id: PatId, f: &mut impl FnMut(ExprId)) {
        self.walk_pats(pat_id, &mut |pat| {
            if let Pat::Expr(expr) | Pat::ConstBlock(expr) = self[pat] {
                f(expr);
            }
        });
    }

    fn binding_hygiene(&self, binding: BindingId) -> HygieneId {
        self.binding_hygiene.get(&binding).copied().unwrap_or(HygieneId::ROOT)
    }

    pub fn expr_path_hygiene(&self, expr: ExprId) -> HygieneId {
        self.expr_hygiene.get(&expr).copied().unwrap_or(HygieneId::ROOT)
    }

    pub fn pat_path_hygiene(&self, pat: PatId) -> HygieneId {
        self.pat_hygiene.get(&pat).copied().unwrap_or(HygieneId::ROOT)
    }

    pub fn expr_or_pat_path_hygiene(&self, id: ExprOrPatId) -> HygieneId {
        match id {
            ExprOrPatId::ExprId(id) => self.expr_path_hygiene(id),
            ExprOrPatId::PatId(id) => self.pat_path_hygiene(id),
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
            self_param: Default::default(),
            binding_hygiene: Default::default(),
            expr_hygiene: Default::default(),
            pat_hygiene: Default::default(),
            types: Default::default(),
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

impl Index<TypeRefId> for Body {
    type Output = TypeRef;

    fn index(&self, b: TypeRefId) -> &TypeRef {
        &self.types[b]
    }
}

// FIXME: Change `node_` prefix to something more reasonable.
// Perhaps `expr_syntax` and `expr_id`?
impl BodySourceMap {
    pub fn expr_or_pat_syntax(&self, id: ExprOrPatId) -> Result<ExprOrPatSource, SyntheticSyntax> {
        match id {
            ExprOrPatId::ExprId(id) => self.expr_syntax(id).map(|it| it.map(AstPtr::wrap_left)),
            ExprOrPatId::PatId(id) => self.pat_syntax(id),
        }
    }

    pub fn expr_syntax(&self, expr: ExprId) -> Result<ExprSource, SyntheticSyntax> {
        self.expr_map_back.get(expr).cloned().ok_or(SyntheticSyntax)
    }

    pub fn node_expr(&self, node: InFile<&ast::Expr>) -> Option<ExprOrPatId> {
        let src = node.map(AstPtr::new);
        self.expr_map.get(&src).cloned()
    }

    pub fn node_macro_file(&self, node: InFile<&ast::MacroCall>) -> Option<MacroFileId> {
        let src = node.map(AstPtr::new);
        self.expansions.get(&src).cloned()
    }

    pub fn macro_calls(
        &self,
    ) -> impl Iterator<Item = (InFile<AstPtr<ast::MacroCall>>, MacroFileId)> + '_ {
        self.expansions.iter().map(|(&a, &b)| (a, b))
    }

    pub fn pat_syntax(&self, pat: PatId) -> Result<ExprOrPatSource, SyntheticSyntax> {
        self.pat_map_back.get(pat).cloned().ok_or(SyntheticSyntax)
    }

    pub fn self_param_syntax(&self) -> Option<InFile<AstPtr<ast::SelfParam>>> {
        self.self_param
    }

    pub fn node_pat(&self, node: InFile<&ast::Pat>) -> Option<PatId> {
        self.pat_map.get(&node.map(AstPtr::new)).cloned()
    }

    pub fn label_syntax(&self, label: LabelId) -> LabelSource {
        self.label_map_back[label]
    }

    pub fn patterns_for_binding(&self, binding: BindingId) -> &[PatId] {
        self.binding_definitions.get(&binding).map_or(&[], Deref::deref)
    }

    pub fn node_label(&self, node: InFile<&ast::Label>) -> Option<LabelId> {
        let src = node.map(AstPtr::new);
        self.label_map.get(&src).cloned()
    }

    pub fn field_syntax(&self, expr: ExprId) -> FieldSource {
        self.field_map_back[&expr]
    }

    pub fn pat_field_syntax(&self, pat: PatId) -> PatFieldSource {
        self.pat_field_map_back[&pat]
    }

    pub fn macro_expansion_expr(&self, node: InFile<&ast::MacroExpr>) -> Option<ExprOrPatId> {
        let src = node.map(AstPtr::new).map(AstPtr::upcast::<ast::MacroExpr>).map(AstPtr::upcast);
        self.expr_map.get(&src).copied()
    }

    pub fn expansions(
        &self,
    ) -> impl Iterator<Item = (&InFile<AstPtr<ast::MacroCall>>, &MacroFileId)> {
        self.expansions.iter()
    }

    pub fn implicit_format_args(
        &self,
        node: InFile<&ast::FormatArgsExpr>,
    ) -> Option<(HygieneId, &[(syntax::TextRange, Name)])> {
        let src = node.map(AstPtr::new).map(AstPtr::upcast::<ast::Expr>);
        let (hygiene, names) =
            self.template_map.as_ref()?.0.get(&self.expr_map.get(&src)?.as_expr()?)?;
        Some((*hygiene, &**names))
    }

    pub fn asm_template_args(
        &self,
        node: InFile<&ast::AsmExpr>,
    ) -> Option<(ExprId, &[Vec<(syntax::TextRange, usize)>])> {
        let src = node.map(AstPtr::new).map(AstPtr::upcast::<ast::Expr>);
        let expr = self.expr_map.get(&src)?.as_expr()?;
        Some(expr).zip(self.template_map.as_ref()?.1.get(&expr).map(std::ops::Deref::deref))
    }

    /// Get a reference to the body source map's diagnostics.
    pub fn diagnostics(&self) -> &[BodyDiagnostic] {
        &self.diagnostics
    }

    fn shrink_to_fit(&mut self) {
        let Self {
            self_param: _,
            expr_map,
            expr_map_back,
            pat_map,
            pat_map_back,
            label_map,
            label_map_back,
            field_map_back,
            pat_field_map_back,
            expansions,
            template_map,
            diagnostics,
            binding_definitions,
            types,
        } = self;
        if let Some(template_map) = template_map {
            template_map.0.shrink_to_fit();
            template_map.1.shrink_to_fit();
        }
        expr_map.shrink_to_fit();
        expr_map_back.shrink_to_fit();
        pat_map.shrink_to_fit();
        pat_map_back.shrink_to_fit();
        label_map.shrink_to_fit();
        label_map_back.shrink_to_fit();
        field_map_back.shrink_to_fit();
        pat_field_map_back.shrink_to_fit();
        expansions.shrink_to_fit();
        diagnostics.shrink_to_fit();
        binding_definitions.shrink_to_fit();
        types.shrink_to_fit();
    }
}
