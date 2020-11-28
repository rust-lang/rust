//! Semantic errors and warnings.
//!
//! The `Diagnostic` trait defines a trait object which can represent any
//! diagnostic.
//!
//! `DiagnosticSink` struct is used as an emitter for diagnostic. When creating
//! a `DiagnosticSink`, you supply a callback which can react to a `dyn
//! Diagnostic` or to any concrete diagnostic (downcasting is used internally).
//!
//! Because diagnostics store file offsets, it's a bad idea to store them
//! directly in salsa. For this reason, every hir subsytem defines it's own
//! strongly-typed closed set of diagnostics which use hir ids internally, are
//! stored in salsa and do *not* implement the `Diagnostic` trait. Instead, a
//! subsystem provides a separate, non-query-based API which can walk all stored
//! values and transform them into instances of `Diagnostic`.

use std::{any::Any, fmt};

use syntax::SyntaxNodePtr;

use crate::InFile;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DiagnosticCode(pub &'static str);

impl DiagnosticCode {
    pub fn as_str(&self) -> &str {
        self.0
    }
}

pub trait Diagnostic: Any + Send + Sync + fmt::Debug + 'static {
    fn code(&self) -> DiagnosticCode;
    fn message(&self) -> String;
    /// Source element that triggered the diagnostics.
    ///
    /// Note that this should reflect "semantics", rather than specific span we
    /// want to highlight. When rendering the diagnostics into an error message,
    /// the IDE will fetch the `SyntaxNode` and will narrow the span
    /// appropriately.
    fn display_source(&self) -> InFile<SyntaxNodePtr>;
    fn as_any(&self) -> &(dyn Any + Send + 'static);
    fn is_experimental(&self) -> bool {
        false
    }
}

pub struct DiagnosticSink<'a> {
    callbacks: Vec<Box<dyn FnMut(&dyn Diagnostic) -> Result<(), ()> + 'a>>,
    filters: Vec<Box<dyn FnMut(&dyn Diagnostic) -> bool + 'a>>,
    default_callback: Box<dyn FnMut(&dyn Diagnostic) + 'a>,
}

impl<'a> DiagnosticSink<'a> {
    pub fn push(&mut self, d: impl Diagnostic) {
        let d: &dyn Diagnostic = &d;
        self._push(d);
    }

    fn _push(&mut self, d: &dyn Diagnostic) {
        for filter in &mut self.filters {
            if !filter(d) {
                return;
            }
        }
        for cb in &mut self.callbacks {
            match cb(d) {
                Ok(()) => return,
                Err(()) => (),
            }
        }
        (self.default_callback)(d)
    }
}

pub struct DiagnosticSinkBuilder<'a> {
    callbacks: Vec<Box<dyn FnMut(&dyn Diagnostic) -> Result<(), ()> + 'a>>,
    filters: Vec<Box<dyn FnMut(&dyn Diagnostic) -> bool + 'a>>,
}

impl<'a> DiagnosticSinkBuilder<'a> {
    pub fn new() -> Self {
        Self { callbacks: Vec::new(), filters: Vec::new() }
    }

    pub fn filter<F: FnMut(&dyn Diagnostic) -> bool + 'a>(mut self, cb: F) -> Self {
        self.filters.push(Box::new(cb));
        self
    }

    pub fn on<D: Diagnostic, F: FnMut(&D) + 'a>(mut self, mut cb: F) -> Self {
        let cb = move |diag: &dyn Diagnostic| match diag.as_any().downcast_ref::<D>() {
            Some(d) => {
                cb(d);
                Ok(())
            }
            None => Err(()),
        };
        self.callbacks.push(Box::new(cb));
        self
    }

    pub fn build<F: FnMut(&dyn Diagnostic) + 'a>(self, default_callback: F) -> DiagnosticSink<'a> {
        DiagnosticSink {
            callbacks: self.callbacks,
            filters: self.filters,
            default_callback: Box::new(default_callback),
        }
    }
}
