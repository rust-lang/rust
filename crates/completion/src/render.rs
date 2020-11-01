//! `render` module provides utilities for rendering completion suggestions
//! into code pieces that will be presented to user.

mod macro_;
mod function;
mod builder_ext;
mod enum_variant;

use hir::HasAttrs;
use ide_db::RootDatabase;
use syntax::TextRange;

use crate::{config::SnippetCap, CompletionContext};

pub(crate) use crate::render::{
    enum_variant::EnumVariantRender, function::FunctionRender, macro_::MacroRender,
};

#[derive(Debug)]
pub(crate) struct RenderContext<'a> {
    completion: &'a CompletionContext<'a>,
}

impl<'a> RenderContext<'a> {
    pub fn new(completion: &'a CompletionContext<'a>) -> RenderContext<'a> {
        RenderContext { completion }
    }

    pub fn snippet_cap(&self) -> Option<SnippetCap> {
        self.completion.config.snippet_cap.clone()
    }

    pub fn db(&self) -> &'a RootDatabase {
        &self.completion.db
    }

    pub fn source_range(&self) -> TextRange {
        self.completion.source_range()
    }

    pub fn is_deprecated(&self, node: impl HasAttrs) -> bool {
        node.attrs(self.db()).by_key("deprecated").exists()
    }
}

impl<'a> From<&'a CompletionContext<'a>> for RenderContext<'a> {
    fn from(ctx: &'a CompletionContext<'a>) -> RenderContext<'a> {
        RenderContext::new(ctx)
    }
}
