//! Defines `Body`: a lowered representation of bodies of functions, statics and
//! consts.
mod lower;
pub mod scope;

use std::{mem, ops::Index, sync::Arc};

use arena::{map::ArenaMap, Arena};
use base_db::CrateId;
use cfg::CfgOptions;
use drop_bomb::DropBomb;
use either::Either;
use hir_expand::{ast_id_map::AstIdMap, hygiene::Hygiene, AstId, HirFileId, InFile, MacroDefId};
use rustc_hash::FxHashMap;
use syntax::{ast, AstNode, AstPtr};
use test_utils::mark;

pub(crate) use lower::LowerCtx;

use crate::{
    attr::Attrs,
    db::DefDatabase,
    expr::{Expr, ExprId, Pat, PatId},
    item_scope::BuiltinShadowMode,
    item_scope::ItemScope,
    nameres::CrateDefMap,
    path::{ModPath, Path},
    src::HasSource,
    AsMacroCall, DefWithBodyId, HasModule, Lookup, ModuleId,
};

/// A subset of Expander that only deals with cfg attributes. We only need it to
/// avoid cyclic queries in crate def map during enum processing.
pub(crate) struct CfgExpander {
    cfg_options: CfgOptions,
    hygiene: Hygiene,
}

pub(crate) struct Expander {
    cfg_expander: CfgExpander,
    crate_def_map: Arc<CrateDefMap>,
    current_file_id: HirFileId,
    ast_id_map: Arc<AstIdMap>,
    module: ModuleId,
    recursion_limit: usize,
}

#[cfg(test)]
const EXPANSION_RECURSION_LIMIT: usize = 32;

#[cfg(not(test))]
const EXPANSION_RECURSION_LIMIT: usize = 128;

impl CfgExpander {
    pub(crate) fn new(
        db: &dyn DefDatabase,
        current_file_id: HirFileId,
        krate: CrateId,
    ) -> CfgExpander {
        let hygiene = Hygiene::new(db.upcast(), current_file_id);
        let cfg_options = db.crate_graph()[krate].cfg_options.clone();
        CfgExpander { cfg_options, hygiene }
    }

    pub(crate) fn parse_attrs(&self, owner: &dyn ast::AttrsOwner) -> Attrs {
        Attrs::new(owner, &self.hygiene)
    }

    pub(crate) fn is_cfg_enabled(&self, owner: &dyn ast::AttrsOwner) -> bool {
        let attrs = self.parse_attrs(owner);
        attrs.is_cfg_enabled(&self.cfg_options)
    }
}

impl Expander {
    pub(crate) fn new(
        db: &dyn DefDatabase,
        current_file_id: HirFileId,
        module: ModuleId,
    ) -> Expander {
        let cfg_expander = CfgExpander::new(db, current_file_id, module.krate);
        let crate_def_map = db.crate_def_map(module.krate);
        let ast_id_map = db.ast_id_map(current_file_id);
        Expander {
            cfg_expander,
            crate_def_map,
            current_file_id,
            ast_id_map,
            module,
            recursion_limit: 0,
        }
    }

    pub(crate) fn enter_expand<T: ast::AstNode>(
        &mut self,
        db: &dyn DefDatabase,
        local_scope: Option<&ItemScope>,
        macro_call: ast::MacroCall,
    ) -> Option<(Mark, T)> {
        self.recursion_limit += 1;
        if self.recursion_limit > EXPANSION_RECURSION_LIMIT {
            mark::hit!(your_stack_belongs_to_me);
            return None;
        }

        let macro_call = InFile::new(self.current_file_id, &macro_call);

        if let Some(call_id) = macro_call.as_call_id(db, self.crate_def_map.krate, |path| {
            if let Some(local_scope) = local_scope {
                if let Some(def) = path.as_ident().and_then(|n| local_scope.get_legacy_macro(n)) {
                    return Some(def);
                }
            }
            self.resolve_path_as_macro(db, &path)
        }) {
            let file_id = call_id.as_file();
            if let Some(node) = db.parse_or_expand(file_id) {
                if let Some(expr) = T::cast(node) {
                    log::debug!("macro expansion {:#?}", expr.syntax());

                    let mark = Mark {
                        file_id: self.current_file_id,
                        ast_id_map: mem::take(&mut self.ast_id_map),
                        bomb: DropBomb::new("expansion mark dropped"),
                    };
                    self.cfg_expander.hygiene = Hygiene::new(db.upcast(), file_id);
                    self.current_file_id = file_id;
                    self.ast_id_map = db.ast_id_map(file_id);
                    return Some((mark, expr));
                }
            }
        }

        // FIXME: Instead of just dropping the error from expansion
        // report it
        None
    }

    pub(crate) fn exit(&mut self, db: &dyn DefDatabase, mut mark: Mark) {
        self.cfg_expander.hygiene = Hygiene::new(db.upcast(), mark.file_id);
        self.current_file_id = mark.file_id;
        self.ast_id_map = mem::take(&mut mark.ast_id_map);
        self.recursion_limit -= 1;
        mark.bomb.defuse();
    }

    pub(crate) fn to_source<T>(&self, value: T) -> InFile<T> {
        InFile { file_id: self.current_file_id, value }
    }

    pub(crate) fn is_cfg_enabled(&self, owner: &dyn ast::AttrsOwner) -> bool {
        self.cfg_expander.is_cfg_enabled(owner)
    }

    fn parse_path(&mut self, path: ast::Path) -> Option<Path> {
        Path::from_src(path, &self.cfg_expander.hygiene)
    }

    fn resolve_path_as_macro(&self, db: &dyn DefDatabase, path: &ModPath) -> Option<MacroDefId> {
        self.crate_def_map
            .resolve_path(db, self.module.local_id, path, BuiltinShadowMode::Other)
            .0
            .take_macros()
    }

    fn ast_id<N: AstNode>(&self, item: &N) -> AstId<N> {
        let file_local_id = self.ast_id_map.ast_id(item);
        AstId::new(self.current_file_id, file_local_id)
    }
}

pub(crate) struct Mark {
    file_id: HirFileId,
    ast_id_map: Arc<AstIdMap>,
    bomb: DropBomb,
}

/// The body of an item (function, const etc.).
#[derive(Debug, Eq, PartialEq)]
pub struct Body {
    pub exprs: Arena<Expr>,
    pub pats: Arena<Pat>,
    /// The patterns for the function's parameters. While the parameter types are
    /// part of the function signature, the patterns are not (they don't change
    /// the external type of the function).
    ///
    /// If this `Body` is for the body of a constant, this will just be
    /// empty.
    pub params: Vec<PatId>,
    /// The `ExprId` of the actual body expression.
    pub body_expr: ExprId,
    pub item_scope: ItemScope,
}

pub type ExprPtr = AstPtr<ast::Expr>;
pub type ExprSource = InFile<ExprPtr>;

pub type PatPtr = Either<AstPtr<ast::Pat>, AstPtr<ast::SelfParam>>;
pub type PatSource = InFile<PatPtr>;

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
    field_map: FxHashMap<(ExprId, usize), InFile<AstPtr<ast::RecordExprField>>>,
    expansions: FxHashMap<InFile<AstPtr<ast::MacroCall>>, HirFileId>,
}

#[derive(Default, Debug, Eq, PartialEq, Clone, Copy)]
pub struct SyntheticSyntax;

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
        let (body, source_map) = Body::new(db, def, expander, params, body);
        (Arc::new(body), Arc::new(source_map))
    }

    pub(crate) fn body_query(db: &dyn DefDatabase, def: DefWithBodyId) -> Arc<Body> {
        db.body_with_source_map(def).0
    }

    fn new(
        db: &dyn DefDatabase,
        def: DefWithBodyId,
        expander: Expander,
        params: Option<ast::ParamList>,
        body: Option<ast::Expr>,
    ) -> (Body, BodySourceMap) {
        lower::lower(db, def, expander, params, body)
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

    pub fn field_syntax(&self, expr: ExprId, field: usize) -> InFile<AstPtr<ast::RecordExprField>> {
        self.field_map[&(expr, field)].clone()
    }
}

#[cfg(test)]
mod tests {
    use base_db::{fixture::WithFixture, SourceDatabase};
    use test_utils::mark;

    use crate::ModuleDefId;

    use super::*;

    fn lower(ra_fixture: &str) -> Arc<Body> {
        let (db, file_id) = crate::test_db::TestDB::with_single_file(ra_fixture);

        let krate = db.crate_graph().iter().next().unwrap();
        let def_map = db.crate_def_map(krate);
        let module = def_map.modules_for_file(file_id).next().unwrap();
        let module = &def_map[module];
        let fn_def = match module.scope.declarations().next().unwrap() {
            ModuleDefId::FunctionId(it) => it,
            _ => panic!(),
        };

        db.body(fn_def.into())
    }

    #[test]
    fn your_stack_belongs_to_me() {
        mark::check!(your_stack_belongs_to_me);
        lower(
            "
macro_rules! n_nuple {
    ($e:tt) => ();
    ($($rest:tt)*) => {{
        (n_nuple!($($rest)*)None,)
    }};
}
fn main() { n_nuple!(1,2,3); }
",
        );
    }
}
