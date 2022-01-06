//! Defines `Body`: a lowered representation of bodies of functions, statics and
//! consts.
mod lower;
#[cfg(test)]
mod tests;
pub mod scope;

use std::{mem, ops::Index, sync::Arc};

use base_db::CrateId;
use cfg::{CfgExpr, CfgOptions};
use drop_bomb::DropBomb;
use either::Either;
use hir_expand::{
    ast_id_map::AstIdMap, hygiene::Hygiene, AstId, ExpandError, ExpandResult, HirFileId, InFile,
    MacroCallId, MacroDefId,
};
use la_arena::{Arena, ArenaMap};
use limit::Limit;
use profile::Count;
use rustc_hash::FxHashMap;
use syntax::{ast, AstNode, AstPtr, SyntaxNodePtr};

use crate::{
    attr::{Attrs, RawAttrs},
    db::DefDatabase,
    expr::{Expr, ExprId, Label, LabelId, Pat, PatId},
    item_scope::BuiltinShadowMode,
    nameres::DefMap,
    path::{ModPath, Path},
    src::HasSource,
    AsMacroCall, BlockId, DefWithBodyId, HasModule, LocalModuleId, Lookup, ModuleId,
    UnresolvedMacro,
};

pub use lower::LowerCtx;

/// A subset of Expander that only deals with cfg attributes. We only need it to
/// avoid cyclic queries in crate def map during enum processing.
#[derive(Debug)]
pub(crate) struct CfgExpander {
    cfg_options: CfgOptions,
    hygiene: Hygiene,
    krate: CrateId,
}

#[derive(Debug)]
pub struct Expander {
    cfg_expander: CfgExpander,
    def_map: Arc<DefMap>,
    current_file_id: HirFileId,
    ast_id_map: Arc<AstIdMap>,
    module: LocalModuleId,
    recursion_limit: usize,
}

#[cfg(test)]
static EXPANSION_RECURSION_LIMIT: Limit = Limit::new(32);

#[cfg(not(test))]
static EXPANSION_RECURSION_LIMIT: Limit = Limit::new(128);

impl CfgExpander {
    pub(crate) fn new(
        db: &dyn DefDatabase,
        current_file_id: HirFileId,
        krate: CrateId,
    ) -> CfgExpander {
        let hygiene = Hygiene::new(db.upcast(), current_file_id);
        let cfg_options = db.crate_graph()[krate].cfg_options.clone();
        CfgExpander { cfg_options, hygiene, krate }
    }

    pub(crate) fn parse_attrs(&self, db: &dyn DefDatabase, owner: &dyn ast::HasAttrs) -> Attrs {
        RawAttrs::new(db, owner, &self.hygiene).filter(db, self.krate)
    }

    pub(crate) fn is_cfg_enabled(&self, db: &dyn DefDatabase, owner: &dyn ast::HasAttrs) -> bool {
        let attrs = self.parse_attrs(db, owner);
        attrs.is_cfg_enabled(&self.cfg_options)
    }
}

impl Expander {
    pub fn new(db: &dyn DefDatabase, current_file_id: HirFileId, module: ModuleId) -> Expander {
        let cfg_expander = CfgExpander::new(db, current_file_id, module.krate);
        let def_map = module.def_map(db);
        let ast_id_map = db.ast_id_map(current_file_id);
        Expander {
            cfg_expander,
            def_map,
            current_file_id,
            ast_id_map,
            module: module.local_id,
            recursion_limit: 0,
        }
    }

    pub fn enter_expand<T: ast::AstNode>(
        &mut self,
        db: &dyn DefDatabase,
        macro_call: ast::MacroCall,
    ) -> Result<ExpandResult<Option<(Mark, T)>>, UnresolvedMacro> {
        if EXPANSION_RECURSION_LIMIT.check(self.recursion_limit + 1).is_err() {
            cov_mark::hit!(your_stack_belongs_to_me);
            return Ok(ExpandResult::str_err(
                "reached recursion limit during macro expansion".into(),
            ));
        }

        let macro_call = InFile::new(self.current_file_id, &macro_call);

        let resolver =
            |path: ModPath| -> Option<MacroDefId> { self.resolve_path_as_macro(db, &path) };

        let mut err = None;
        let call_id =
            macro_call.as_call_id_with_errors(db, self.def_map.krate(), resolver, &mut |e| {
                err.get_or_insert(e);
            })?;
        let call_id = match call_id {
            Ok(it) => it,
            Err(_) => {
                return Ok(ExpandResult { value: None, err });
            }
        };

        Ok(self.enter_expand_inner(db, call_id, err))
    }

    pub fn enter_expand_id<T: ast::AstNode>(
        &mut self,
        db: &dyn DefDatabase,
        call_id: MacroCallId,
    ) -> ExpandResult<Option<(Mark, T)>> {
        self.enter_expand_inner(db, call_id, None)
    }

    fn enter_expand_inner<T: ast::AstNode>(
        &mut self,
        db: &dyn DefDatabase,
        call_id: MacroCallId,
        mut err: Option<ExpandError>,
    ) -> ExpandResult<Option<(Mark, T)>> {
        if err.is_none() {
            err = db.macro_expand_error(call_id);
        }

        let file_id = call_id.as_file();

        let raw_node = match db.parse_or_expand(file_id) {
            Some(it) => it,
            None => {
                // Only `None` if the macro expansion produced no usable AST.
                if err.is_none() {
                    tracing::warn!("no error despite `parse_or_expand` failing");
                }

                return ExpandResult::only_err(err.unwrap_or_else(|| {
                    mbe::ExpandError::Other("failed to parse macro invocation".into())
                }));
            }
        };

        let node = match T::cast(raw_node) {
            Some(it) => it,
            None => {
                // This can happen without being an error, so only forward previous errors.
                return ExpandResult { value: None, err };
            }
        };

        tracing::debug!("macro expansion {:#?}", node.syntax());

        self.recursion_limit += 1;
        let mark = Mark {
            file_id: self.current_file_id,
            ast_id_map: mem::take(&mut self.ast_id_map),
            bomb: DropBomb::new("expansion mark dropped"),
        };
        self.cfg_expander.hygiene = Hygiene::new(db.upcast(), file_id);
        self.current_file_id = file_id;
        self.ast_id_map = db.ast_id_map(file_id);

        ExpandResult { value: Some((mark, node)), err }
    }

    pub fn exit(&mut self, db: &dyn DefDatabase, mut mark: Mark) {
        self.cfg_expander.hygiene = Hygiene::new(db.upcast(), mark.file_id);
        self.current_file_id = mark.file_id;
        self.ast_id_map = mem::take(&mut mark.ast_id_map);
        self.recursion_limit -= 1;
        mark.bomb.defuse();
    }

    pub(crate) fn to_source<T>(&self, value: T) -> InFile<T> {
        InFile { file_id: self.current_file_id, value }
    }

    pub(crate) fn parse_attrs(&self, db: &dyn DefDatabase, owner: &dyn ast::HasAttrs) -> Attrs {
        self.cfg_expander.parse_attrs(db, owner)
    }

    pub(crate) fn cfg_options(&self) -> &CfgOptions {
        &self.cfg_expander.cfg_options
    }

    pub fn current_file_id(&self) -> HirFileId {
        self.current_file_id
    }

    fn parse_path(&mut self, db: &dyn DefDatabase, path: ast::Path) -> Option<Path> {
        let ctx = LowerCtx::with_hygiene(db, &self.cfg_expander.hygiene);
        Path::from_src(path, &ctx)
    }

    fn resolve_path_as_macro(&self, db: &dyn DefDatabase, path: &ModPath) -> Option<MacroDefId> {
        self.def_map.resolve_path(db, self.module, path, BuiltinShadowMode::Other).0.take_macros()
    }

    fn ast_id<N: AstNode>(&self, item: &N) -> AstId<N> {
        let file_local_id = self.ast_id_map.ast_id(item);
        AstId::new(self.current_file_id, file_local_id)
    }
}

#[derive(Debug)]
pub struct Mark {
    file_id: HirFileId,
    ast_id_map: Arc<AstIdMap>,
    bomb: DropBomb,
}

/// The body of an item (function, const etc.).
#[derive(Debug, Eq, PartialEq)]
pub struct Body {
    pub exprs: Arena<Expr>,
    pub pats: Arena<Pat>,
    pub labels: Arena<Label>,
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
    expr_map_back: ArenaMap<ExprId, Result<ExprSource, SyntheticSyntax>>,

    pat_map: FxHashMap<PatSource, PatId>,
    pat_map_back: ArenaMap<PatId, Result<PatSource, SyntheticSyntax>>,

    label_map: FxHashMap<LabelSource, LabelId>,
    label_map_back: ArenaMap<LabelId, LabelSource>,

    /// We don't create explicit nodes for record fields (`S { record_field: 92 }`).
    /// Instead, we use id of expression (`92`) to identify the field.
    field_map: FxHashMap<InFile<AstPtr<ast::RecordExprField>>, ExprId>,
    field_map_back: FxHashMap<ExprId, InFile<AstPtr<ast::RecordExprField>>>,

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
    UnresolvedProcMacro { node: InFile<AstPtr<ast::MacroCall>> },
    UnresolvedMacroCall { node: InFile<AstPtr<ast::MacroCall>>, path: ModPath },
}

impl Body {
    pub(crate) fn body_with_source_map_query(
        db: &dyn DefDatabase,
        def: DefWithBodyId,
    ) -> (Arc<Body>, Arc<BodySourceMap>) {
        let _p = profile::span("body_with_source_map_query");
        let mut params = None;

        let (file_id, module, body) = match def {
            DefWithBodyId::FunctionId(f) => {
                let f = f.lookup(db);
                let src = f.source(db);
                params = src.value.param_list();
                (src.file_id, f.module(db), src.value.body().map(ast::Expr::from))
            }
            DefWithBodyId::ConstId(c) => {
                let c = c.lookup(db);
                let src = c.source(db);
                (src.file_id, c.module(db), src.value.body())
            }
            DefWithBodyId::StaticId(s) => {
                let s = s.lookup(db);
                let src = s.source(db);
                (src.file_id, s.module(db), src.value.body())
            }
        };
        let expander = Expander::new(db, file_id, module);
        let (mut body, source_map) = Body::new(db, expander, params, body);
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
        self.block_scopes
            .iter()
            .map(move |block| (*block, db.block_def_map(*block).expect("block ID without DefMap")))
    }

    fn new(
        db: &dyn DefDatabase,
        expander: Expander,
        params: Option<ast::ParamList>,
        body: Option<ast::Expr>,
    ) -> (Body, BodySourceMap) {
        lower::lower(db, expander, params, body)
    }

    fn shrink_to_fit(&mut self) {
        let Self { _c: _, body_expr: _, block_scopes, exprs, labels, params, pats } = self;
        block_scopes.shrink_to_fit();
        exprs.shrink_to_fit();
        labels.shrink_to_fit();
        params.shrink_to_fit();
        pats.shrink_to_fit();
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

// FIXME: Change `node_` prefix to something more reasonable.
// Perhaps `expr_syntax` and `expr_id`?
impl BodySourceMap {
    pub fn expr_syntax(&self, expr: ExprId) -> Result<ExprSource, SyntheticSyntax> {
        self.expr_map_back[expr].clone()
    }

    pub fn node_expr(&self, node: InFile<&ast::Expr>) -> Option<ExprId> {
        let src = node.map(|it| AstPtr::new(it));
        self.expr_map.get(&src).cloned()
    }

    pub fn node_macro_file(&self, node: InFile<&ast::MacroCall>) -> Option<HirFileId> {
        let src = node.map(|it| AstPtr::new(it));
        self.expansions.get(&src).cloned()
    }

    pub fn pat_syntax(&self, pat: PatId) -> Result<PatSource, SyntheticSyntax> {
        self.pat_map_back[pat].clone()
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
        let src = node.map(|it| AstPtr::new(it));
        self.label_map.get(&src).cloned()
    }

    pub fn field_syntax(&self, expr: ExprId) -> InFile<AstPtr<ast::RecordExprField>> {
        self.field_map_back[&expr].clone()
    }
    pub fn node_field(&self, node: InFile<&ast::RecordExprField>) -> Option<ExprId> {
        let src = node.map(|it| AstPtr::new(it));
        self.field_map.get(&src).cloned()
    }

    /// Get a reference to the body source map's diagnostics.
    pub fn diagnostics(&self) -> &[BodyDiagnostic] {
        &self.diagnostics
    }
}
