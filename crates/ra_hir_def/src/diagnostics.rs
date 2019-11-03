//! Diagnostics produced by `hir_def`.

use std::any::Any;

use hir_expand::diagnostics::Diagnostic;
use ra_syntax::{ast, AstPtr, SyntaxNodePtr};
use relative_path::RelativePathBuf;

use hir_expand::{HirFileId, Source};

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
    fn source(&self) -> Source<SyntaxNodePtr> {
        Source { file_id: self.file, ast: self.decl.into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}
