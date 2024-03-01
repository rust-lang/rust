//! Context for lowering paths.
use std::cell::OnceCell;

use hir_expand::{
    span_map::{SpanMap, SpanMapRef},
    AstId, HirFileId, InFile,
};
use span::{AstIdMap, AstIdNode};
use syntax::ast;
use triomphe::Arc;

use crate::{db::DefDatabase, path::Path};

pub struct LowerCtx<'a> {
    pub db: &'a dyn DefDatabase,
    span_map: SpanMap,
    // FIXME: This optimization is probably pointless, ast id map should pretty much always exist anyways.
    ast_id_map: Option<(HirFileId, OnceCell<Arc<AstIdMap>>)>,
}

impl<'a> LowerCtx<'a> {
    pub fn new(db: &'a dyn DefDatabase, span_map: SpanMap, file_id: HirFileId) -> Self {
        LowerCtx { db, span_map, ast_id_map: Some((file_id, OnceCell::new())) }
    }

    pub fn with_file_id(db: &'a dyn DefDatabase, file_id: HirFileId) -> Self {
        LowerCtx {
            db,
            span_map: db.span_map(file_id),
            ast_id_map: Some((file_id, OnceCell::new())),
        }
    }

    pub fn with_span_map(db: &'a dyn DefDatabase, span_map: SpanMap) -> Self {
        LowerCtx { db, span_map, ast_id_map: None }
    }

    pub(crate) fn span_map(&self) -> SpanMapRef<'_> {
        self.span_map.as_ref()
    }

    pub(crate) fn lower_path(&self, ast: ast::Path) -> Option<Path> {
        Path::from_src(self, ast)
    }

    pub(crate) fn ast_id<N: AstIdNode>(&self, item: &N) -> Option<AstId<N>> {
        let &(file_id, ref ast_id_map) = self.ast_id_map.as_ref()?;
        let ast_id_map = ast_id_map.get_or_init(|| self.db.ast_id_map(file_id));
        Some(InFile::new(file_id, ast_id_map.ast_id(item)))
    }
}
