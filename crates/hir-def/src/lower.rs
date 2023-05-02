//! Context for lowering paths.
use hir_expand::{ast_id_map::AstIdMap, hygiene::Hygiene, AstId, HirFileId, InFile};
use once_cell::unsync::OnceCell;
use syntax::ast;
use triomphe::Arc;

use crate::{db::DefDatabase, path::Path};

pub struct LowerCtx<'a> {
    pub db: &'a dyn DefDatabase,
    hygiene: Hygiene,
    ast_id_map: Option<(HirFileId, OnceCell<Arc<AstIdMap>>)>,
}

impl<'a> LowerCtx<'a> {
    pub fn new(db: &'a dyn DefDatabase, hygiene: &Hygiene, file_id: HirFileId) -> Self {
        LowerCtx { db, hygiene: hygiene.clone(), ast_id_map: Some((file_id, OnceCell::new())) }
    }

    pub fn with_file_id(db: &'a dyn DefDatabase, file_id: HirFileId) -> Self {
        LowerCtx {
            db,
            hygiene: Hygiene::new(db.upcast(), file_id),
            ast_id_map: Some((file_id, OnceCell::new())),
        }
    }

    pub fn with_hygiene(db: &'a dyn DefDatabase, hygiene: &Hygiene) -> Self {
        LowerCtx { db, hygiene: hygiene.clone(), ast_id_map: None }
    }

    pub(crate) fn hygiene(&self) -> &Hygiene {
        &self.hygiene
    }

    pub(crate) fn lower_path(&self, ast: ast::Path) -> Option<Path> {
        Path::from_src(ast, self)
    }

    pub(crate) fn ast_id<N: syntax::AstNode>(&self, item: &N) -> Option<AstId<N>> {
        let &(file_id, ref ast_id_map) = self.ast_id_map.as_ref()?;
        let ast_id_map = ast_id_map.get_or_init(|| self.db.ast_id_map(file_id));
        Some(InFile::new(file_id, ast_id_map.ast_id(item)))
    }
}
