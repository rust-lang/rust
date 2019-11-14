//! FIXME: write short doc here
mod lower;
pub mod scope;

use std::{ops::Index, sync::Arc};

use hir_expand::{
    either::Either, hygiene::Hygiene, AstId, HirFileId, MacroCallLoc, MacroDefId, MacroFileKind,
    Source,
};
use ra_arena::{map::ArenaMap, Arena};
use ra_syntax::{ast, AstNode, AstPtr};
use rustc_hash::FxHashMap;

use crate::{
    db::DefDatabase2,
    expr::{Expr, ExprId, Pat, PatId},
    nameres::CrateDefMap,
    path::Path,
    AstItemDef, DefWithBodyId, ModuleId,
};

pub struct Expander {
    crate_def_map: Arc<CrateDefMap>,
    current_file_id: HirFileId,
    hygiene: Hygiene,
    module: ModuleId,
}

impl Expander {
    pub fn new(db: &impl DefDatabase2, current_file_id: HirFileId, module: ModuleId) -> Expander {
        let crate_def_map = db.crate_def_map(module.krate);
        let hygiene = Hygiene::new(db, current_file_id);
        Expander { crate_def_map, current_file_id, hygiene, module }
    }

    fn enter_expand(
        &mut self,
        db: &impl DefDatabase2,
        macro_call: ast::MacroCall,
    ) -> Option<(Mark, ast::Expr)> {
        let ast_id = AstId::new(
            self.current_file_id,
            db.ast_id_map(self.current_file_id).ast_id(&macro_call),
        );

        if let Some(path) = macro_call.path().and_then(|path| self.parse_path(path)) {
            if let Some(def) = self.resolve_path_as_macro(db, &path) {
                let call_id = db.intern_macro(MacroCallLoc { def, ast_id });
                let file_id = call_id.as_file(MacroFileKind::Expr);
                if let Some(node) = db.parse_or_expand(file_id) {
                    if let Some(expr) = ast::Expr::cast(node) {
                        log::debug!("macro expansion {:#?}", expr.syntax());

                        let mark = Mark { file_id: self.current_file_id };
                        self.hygiene = Hygiene::new(db, file_id);
                        self.current_file_id = file_id;

                        return Some((mark, expr));
                    }
                }
            }
        }

        // FIXME: Instead of just dropping the error from expansion
        // report it
        None
    }

    fn exit(&mut self, db: &impl DefDatabase2, mark: Mark) {
        self.hygiene = Hygiene::new(db, mark.file_id);
        self.current_file_id = mark.file_id;
        std::mem::forget(mark);
    }

    fn to_source<T>(&self, ast: T) -> Source<T> {
        Source { file_id: self.current_file_id, ast }
    }

    fn parse_path(&mut self, path: ast::Path) -> Option<Path> {
        Path::from_src(path, &self.hygiene)
    }

    fn resolve_path_as_macro(&self, db: &impl DefDatabase2, path: &Path) -> Option<MacroDefId> {
        self.crate_def_map.resolve_path(db, self.module.module_id, path).0.get_macros()
    }
}

struct Mark {
    file_id: HirFileId,
}

impl Drop for Mark {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            panic!("dropped mark")
        }
    }
}

/// The body of an item (function, const etc.).
#[derive(Debug, Eq, PartialEq)]
pub struct Body {
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

pub type ExprPtr = Either<AstPtr<ast::Expr>, AstPtr<ast::RecordField>>;
pub type ExprSource = Source<ExprPtr>;

pub type PatPtr = Either<AstPtr<ast::Pat>, AstPtr<ast::SelfParam>>;
pub type PatSource = Source<PatPtr>;

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
    field_map: FxHashMap<(ExprId, usize), AstPtr<ast::RecordField>>,
}

impl Body {
    pub(crate) fn body_with_source_map_query(
        db: &impl DefDatabase2,
        def: DefWithBodyId,
    ) -> (Arc<Body>, Arc<BodySourceMap>) {
        let mut params = None;

        let (file_id, module, body) = match def {
            DefWithBodyId::FunctionId(f) => {
                let src = f.source(db);
                params = src.ast.param_list();
                (src.file_id, f.module(db), src.ast.body().map(ast::Expr::from))
            }
            DefWithBodyId::ConstId(c) => {
                let src = c.source(db);
                (src.file_id, c.module(db), src.ast.body())
            }
            DefWithBodyId::StaticId(s) => {
                let src = s.source(db);
                (src.file_id, s.module(db), src.ast.body())
            }
        };
        let expander = Expander::new(db, file_id, module);
        let (body, source_map) = Body::new(db, expander, params, body);
        (Arc::new(body), Arc::new(source_map))
    }

    pub(crate) fn body_query(db: &impl DefDatabase2, def: DefWithBodyId) -> Arc<Body> {
        db.body_with_source_map(def).0
    }

    fn new(
        db: &impl DefDatabase2,
        expander: Expander,
        params: Option<ast::ParamList>,
        body: Option<ast::Expr>,
    ) -> (Body, BodySourceMap) {
        lower::lower(db, expander, params, body)
    }

    pub fn params(&self) -> &[PatId] {
        &self.params
    }

    pub fn body_expr(&self) -> ExprId {
        self.body_expr
    }

    pub fn exprs(&self) -> impl Iterator<Item = (ExprId, &Expr)> {
        self.exprs.iter()
    }

    pub fn pats(&self) -> impl Iterator<Item = (PatId, &Pat)> {
        self.pats.iter()
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
    pub fn expr_syntax(&self, expr: ExprId) -> Option<ExprSource> {
        self.expr_map_back.get(expr).copied()
    }

    pub fn node_expr(&self, node: Source<&ast::Expr>) -> Option<ExprId> {
        let src = node.map(|it| Either::A(AstPtr::new(it)));
        self.expr_map.get(&src).cloned()
    }

    pub fn pat_syntax(&self, pat: PatId) -> Option<PatSource> {
        self.pat_map_back.get(pat).copied()
    }

    pub fn node_pat(&self, node: Source<&ast::Pat>) -> Option<PatId> {
        let src = node.map(|it| Either::A(AstPtr::new(it)));
        self.pat_map.get(&src).cloned()
    }

    pub fn field_syntax(&self, expr: ExprId, field: usize) -> AstPtr<ast::RecordField> {
        self.field_map[&(expr, field)]
    }
}
