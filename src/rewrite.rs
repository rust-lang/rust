// A generic trait to abstract the rewriting of an element (of the AST).

use std::cell::RefCell;

use syntax::parse::ParseSess;
use syntax::ptr;
use syntax::source_map::{SourceMap, Span};

use crate::config::{Config, IndentStyle};
use crate::shape::Shape;
use crate::visitor::SnippetProvider;
use crate::FormatReport;

pub trait Rewrite {
    /// Rewrite self into shape.
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String>;
}

impl<T: Rewrite> Rewrite for ptr::P<T> {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        (**self).rewrite(context, shape)
    }
}

#[derive(Clone)]
pub struct RewriteContext<'a> {
    pub parse_session: &'a ParseSess,
    pub source_map: &'a SourceMap,
    pub config: &'a Config,
    pub inside_macro: RefCell<bool>,
    // Force block indent style even if we are using visual indent style.
    pub use_block: RefCell<bool>,
    // When `is_if_else_block` is true, unindent the comment on top
    // of the `else` or `else if`.
    pub is_if_else_block: RefCell<bool>,
    // When rewriting chain, veto going multi line except the last element
    pub force_one_line_chain: RefCell<bool>,
    pub snippet_provider: &'a SnippetProvider<'a>,
    // Used for `format_snippet`
    pub(crate) macro_rewrite_failure: RefCell<bool>,
    pub(crate) report: FormatReport,
    pub skip_macro_names: RefCell<Vec<String>>,
}

impl<'a> RewriteContext<'a> {
    pub fn snippet(&self, span: Span) -> &str {
        self.snippet_provider.span_to_snippet(span).unwrap()
    }

    /// Returns `true` if we should use block indent style for rewriting function call.
    pub fn use_block_indent(&self) -> bool {
        self.config.indent_style() == IndentStyle::Block || *self.use_block.borrow()
    }

    pub fn budget(&self, used_width: usize) -> usize {
        self.config.max_width().saturating_sub(used_width)
    }

    pub fn inside_macro(&self) -> bool {
        *self.inside_macro.borrow()
    }

    pub fn is_if_else_block(&self) -> bool {
        *self.is_if_else_block.borrow()
    }
}
