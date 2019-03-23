use std::{fmt, any::Any};

use ra_syntax::{SyntaxNodePtr, AstPtr, ast};

use crate::HirFileId;
use relative_path::RelativePathBuf;

/// Diagnostic defines hir API for errors and warnings.
///
/// It is used as a `dyn` object, which you can downcast to a concrete
/// diagnostic. Diagnostics are structured, meaning that they include rich
/// information which can be used by IDE to create fixes. Diagnostics are
/// expressed in terms of macro-expanded syntax tree nodes (so, it's a bad idea
/// to diagnostic in a salsa value).
///
/// Internally, various subsystems of hir produce diagnostics specific to a
/// subsytem (typically, an `enum`), which are safe to store in salsa but do not
/// include source locations. Such internal diagnostic are transformed into an
/// instance of `Diagnostic` on demand.
pub trait Diagnostic: Any + Send + Sync + fmt::Debug + 'static {
    fn file(&self) -> HirFileId;
    fn syntax_node(&self) -> SyntaxNodePtr;
    fn message(&self) -> String;
    fn as_any(&self) -> &(Any + Send + 'static);
}

impl dyn Diagnostic {
    pub fn downcast_ref<D: Diagnostic>(&self) -> Option<&D> {
        self.as_any().downcast_ref()
    }
}

#[derive(Debug, Default)]
pub struct Diagnostics {
    data: Vec<Box<dyn Diagnostic>>,
}

impl Diagnostics {
    pub fn push(&mut self, d: impl Diagnostic) {
        self.data.push(Box::new(d))
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a dyn Diagnostic> + 'a {
        self.data.iter().map(|it| it.as_ref())
    }
}

#[derive(Debug)]
pub struct NoSuchField {
    pub file: HirFileId,
    pub field: AstPtr<ast::NamedField>,
}

impl Diagnostic for NoSuchField {
    fn file(&self) -> HirFileId {
        self.file
    }
    fn syntax_node(&self) -> SyntaxNodePtr {
        self.field.into()
    }
    fn message(&self) -> String {
        "no such field".to_string()
    }
    fn as_any(&self) -> &(Any + Send + 'static) {
        self
    }
}

#[derive(Debug)]
pub struct UnresolvedModule {
    pub file: HirFileId,
    pub decl: AstPtr<ast::Module>,
    pub candidate: RelativePathBuf,
}

impl Diagnostic for UnresolvedModule {
    fn file(&self) -> HirFileId {
        self.file
    }
    fn syntax_node(&self) -> SyntaxNodePtr {
        self.decl.into()
    }
    fn message(&self) -> String {
        "unresolved module".to_string()
    }
    fn as_any(&self) -> &(Any + Send + 'static) {
        self
    }
}
