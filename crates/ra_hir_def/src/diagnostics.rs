//! Diagnostics produced by `hir_def`.

use std::any::Any;

use hir_expand::diagnostics::Diagnostic;
use ra_db::RelativePathBuf;
use ra_syntax::{ast, AstPtr, SyntaxNodePtr};

use hir_expand::{HirFileId, InFile};

#[derive(Debug)]
pub struct UnresolvedModule {
    pub file: HirFileId,
    pub decl: AstPtr<ast::Module>,
    pub candidate: RelativePathBuf,
}

impl Diagnostic for UnresolvedModule {
    fn message(&self) -> String {
        "unresolved module".to_string()
    }
    fn source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.decl.into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}
