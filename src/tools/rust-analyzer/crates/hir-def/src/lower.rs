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
    file_id: HirFileId,
    span_map: OnceCell<SpanMap>,
    ast_id_map: OnceCell<Arc<AstIdMap>>,
}

impl<'a> LowerCtx<'a> {
    pub fn new(db: &'a dyn DefDatabase, file_id: HirFileId) -> Self {
        LowerCtx { db, file_id, span_map: OnceCell::new(), ast_id_map: OnceCell::new() }
    }

    pub fn with_span_map_cell(
        db: &'a dyn DefDatabase,
        file_id: HirFileId,
        span_map: OnceCell<SpanMap>,
    ) -> Self {
        LowerCtx { db, file_id, span_map, ast_id_map: OnceCell::new() }
    }

    pub(crate) fn span_map(&self) -> SpanMapRef<'_> {
        self.span_map.get_or_init(|| self.db.span_map(self.file_id)).as_ref()
    }

    pub(crate) fn lower_path(&self, ast: ast::Path) -> Option<Path> {
        Path::from_src(self, ast)
    }

    pub(crate) fn ast_id<N: AstIdNode>(&self, item: &N) -> AstId<N> {
        InFile::new(
            self.file_id,
            self.ast_id_map.get_or_init(|| self.db.ast_id_map(self.file_id)).ast_id(item),
        )
    }
}
