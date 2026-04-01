use std::mem;
use std::sync::{Arc, OnceLock};

use rustc_ast::{Attribute, Crate};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Span;

#[derive(Clone, Default)]
pub struct AttrStorage(pub Arc<OnceLock<Vec<Span>>>);

pub struct AttrCollector {
    storage: AttrStorage,
    attrs: Vec<Span>,
}

impl AttrCollector {
    pub fn new(storage: AttrStorage) -> Self {
        Self {
            storage,
            attrs: Vec::new(),
        }
    }
}

impl_lint_pass!(AttrCollector => []);

impl EarlyLintPass for AttrCollector {
    fn check_attribute(&mut self, _cx: &EarlyContext<'_>, attr: &Attribute) {
        self.attrs.push(attr.span);
    }

    fn check_crate_post(&mut self, _: &EarlyContext<'_>, _: &Crate) {
        self.storage
            .0
            .set(mem::take(&mut self.attrs))
            .expect("should only be called once");
    }
}
