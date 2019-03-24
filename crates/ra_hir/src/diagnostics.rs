use std::{fmt, any::Any};

use ra_syntax::{SyntaxNodePtr, AstPtr, ast};
use relative_path::RelativePathBuf;

use crate::HirFileId;

/// Diagnostic defines hir API for errors and warnings.
///
/// It is used as a `dyn` object, which you can downcast to a concrete
/// diagnostic. DiagnosticSink are structured, meaning that they include rich
/// information which can be used by IDE to create fixes. DiagnosticSink are
/// expressed in terms of macro-expanded syntax tree nodes (so, it's a bad idea
/// to diagnostic in a salsa value).
///
/// Internally, various subsystems of hir produce diagnostics specific to a
/// subsystem (typically, an `enum`), which are safe to store in salsa but do not
/// include source locations. Such internal diagnostic are transformed into an
/// instance of `Diagnostic` on demand.
pub trait Diagnostic: Any + Send + Sync + fmt::Debug + 'static {
    fn message(&self) -> String;
    fn file(&self) -> HirFileId;
    fn syntax_node(&self) -> SyntaxNodePtr;
    fn as_any(&self) -> &(dyn Any + Send + 'static);
}

impl dyn Diagnostic {
    pub fn downcast_ref<D: Diagnostic>(&self) -> Option<&D> {
        self.as_any().downcast_ref()
    }
}

pub struct DiagnosticSink<'a> {
    callbacks: Vec<Box<dyn FnMut(&dyn Diagnostic) -> Result<(), ()> + 'a>>,
    default_callback: Box<dyn FnMut(&dyn Diagnostic) + 'a>,
}

impl<'a> DiagnosticSink<'a> {
    pub fn new(cb: impl FnMut(&dyn Diagnostic) + 'a) -> DiagnosticSink<'a> {
        DiagnosticSink { callbacks: Vec::new(), default_callback: Box::new(cb) }
    }

    pub fn on<D: Diagnostic, F: FnMut(&D) + 'a>(mut self, mut cb: F) -> DiagnosticSink<'a> {
        let cb = move |diag: &dyn Diagnostic| match diag.downcast_ref::<D>() {
            Some(d) => {
                cb(d);
                Ok(())
            }
            None => Err(()),
        };
        self.callbacks.push(Box::new(cb));
        self
    }

    pub(crate) fn push(&mut self, d: impl Diagnostic) {
        let d: &dyn Diagnostic = &d;
        for cb in self.callbacks.iter_mut() {
            match cb(d) {
                Ok(()) => return,
                Err(()) => (),
            }
        }
        (self.default_callback)(d)
    }
}

#[derive(Debug)]
pub struct NoSuchField {
    pub file: HirFileId,
    pub field: AstPtr<ast::NamedField>,
}

impl Diagnostic for NoSuchField {
    fn message(&self) -> String {
        "no such field".to_string()
    }
    fn file(&self) -> HirFileId {
        self.file
    }
    fn syntax_node(&self) -> SyntaxNodePtr {
        self.field.into()
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
    fn message(&self) -> String {
        "unresolved module".to_string()
    }
    fn file(&self) -> HirFileId {
        self.file
    }
    fn syntax_node(&self) -> SyntaxNodePtr {
        self.decl.into()
    }
    fn as_any(&self) -> &(Any + Send + 'static) {
        self
    }
}
