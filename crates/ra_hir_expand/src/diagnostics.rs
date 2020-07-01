//! Semantic errors and warnings.
//!
//! The `Diagnostic` trait defines a trait object which can represent any
//! diagnostic.
//!
//! `DiagnosticSink` struct is used as an emitter for diagnostic. When creating
//! a `DiagnosticSink`, you supply a callback which can react to a `dyn
//! Diagnostic` or to any concrete diagnostic (downcasting is sued internally).
//!
//! Because diagnostics store file offsets, it's a bad idea to store them
//! directly in salsa. For this reason, every hir subsytem defines it's own
//! strongly-typed closed set of diagnostics which use hir ids internally, are
//! stored in salsa and do *not* implement the `Diagnostic` trait. Instead, a
//! subsystem provides a separate, non-query-based API which can walk all stored
//! values and transform them into instances of `Diagnostic`.

use std::{any::Any, fmt};

use ra_syntax::{SyntaxNode, SyntaxNodePtr};

use crate::{db::AstDatabase, InFile};

pub trait Diagnostic: Any + Send + Sync + fmt::Debug + 'static {
    fn message(&self) -> String;
    fn source(&self) -> InFile<SyntaxNodePtr>;
    fn as_any(&self) -> &(dyn Any + Send + 'static);
}

pub trait AstDiagnostic {
    type AST;
    fn ast(&self, db: &dyn AstDatabase) -> Self::AST;
}

impl dyn Diagnostic {
    pub fn syntax_node(&self, db: &impl AstDatabase) -> SyntaxNode {
        let node = db.parse_or_expand(self.source().file_id).unwrap();
        self.source().value.to_node(&node)
    }

    pub fn downcast_ref<D: Diagnostic>(&self) -> Option<&D> {
        self.as_any().downcast_ref()
    }
}

pub struct DiagnosticSink<'a> {
    callbacks: Vec<Box<dyn FnMut(&dyn Diagnostic) -> Result<(), ()> + 'a>>,
    default_callback: Box<dyn FnMut(&dyn Diagnostic) + 'a>,
}

impl<'a> DiagnosticSink<'a> {
    /// FIXME: split `new` and `on` into a separate builder type
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

    pub fn push(&mut self, d: impl Diagnostic) {
        let d: &dyn Diagnostic = &d;
        self._push(d);
    }

    fn _push(&mut self, d: &dyn Diagnostic) {
        for cb in self.callbacks.iter_mut() {
            match cb(d) {
                Ok(()) => return,
                Err(()) => (),
            }
        }
        (self.default_callback)(d)
    }
}
