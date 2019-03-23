use std::{fmt, any::Any};

use ra_syntax::{SyntaxNodePtr, AstPtr, ast};

use crate::HirFileId;

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
    pub(crate) file: HirFileId,
    pub(crate) field: AstPtr<ast::NamedField>,
}

impl NoSuchField {
    pub fn field(&self) -> AstPtr<ast::NamedField> {
        self.field
    }
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
