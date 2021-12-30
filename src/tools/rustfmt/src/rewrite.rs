// A generic trait to abstract the rewriting of an element (of the AST).

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use rustc_ast::ptr;
use rustc_span::Span;

use crate::config::{Config, IndentStyle};
use crate::parse::session::ParseSess;
use crate::shape::Shape;
use crate::skip::SkipContext;
use crate::visitor::SnippetProvider;
use crate::FormatReport;

pub(crate) trait Rewrite {
    /// Rewrite self into shape.
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String>;
}

impl<T: Rewrite> Rewrite for ptr::P<T> {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        (**self).rewrite(context, shape)
    }
}

#[derive(Clone)]
pub(crate) struct RewriteContext<'a> {
    pub(crate) parse_sess: &'a ParseSess,
    pub(crate) config: &'a Config,
    pub(crate) inside_macro: Rc<Cell<bool>>,
    // Force block indent style even if we are using visual indent style.
    pub(crate) use_block: Cell<bool>,
    // When `is_if_else_block` is true, unindent the comment on top
    // of the `else` or `else if`.
    pub(crate) is_if_else_block: Cell<bool>,
    // When rewriting chain, veto going multi line except the last element
    pub(crate) force_one_line_chain: Cell<bool>,
    pub(crate) snippet_provider: &'a SnippetProvider,
    // Used for `format_snippet`
    pub(crate) macro_rewrite_failure: Cell<bool>,
    pub(crate) is_macro_def: bool,
    pub(crate) report: FormatReport,
    pub(crate) skip_context: SkipContext,
    pub(crate) skipped_range: Rc<RefCell<Vec<(usize, usize)>>>,
}

pub(crate) struct InsideMacroGuard {
    is_nested_macro_context: bool,
    inside_macro_ref: Rc<Cell<bool>>,
}

impl InsideMacroGuard {
    pub(crate) fn is_nested(&self) -> bool {
        self.is_nested_macro_context
    }
}

impl Drop for InsideMacroGuard {
    fn drop(&mut self) {
        self.inside_macro_ref.replace(self.is_nested_macro_context);
    }
}

impl<'a> RewriteContext<'a> {
    pub(crate) fn snippet(&self, span: Span) -> &str {
        self.snippet_provider.span_to_snippet(span).unwrap()
    }

    /// Returns `true` if we should use block indent style for rewriting function call.
    pub(crate) fn use_block_indent(&self) -> bool {
        self.config.indent_style() == IndentStyle::Block || self.use_block.get()
    }

    pub(crate) fn budget(&self, used_width: usize) -> usize {
        self.config.max_width().saturating_sub(used_width)
    }

    pub(crate) fn inside_macro(&self) -> bool {
        self.inside_macro.get()
    }

    pub(crate) fn enter_macro(&self) -> InsideMacroGuard {
        let is_nested_macro_context = self.inside_macro.replace(true);
        InsideMacroGuard {
            is_nested_macro_context,
            inside_macro_ref: self.inside_macro.clone(),
        }
    }

    pub(crate) fn leave_macro(&self) {
        self.inside_macro.replace(false);
    }

    pub(crate) fn is_if_else_block(&self) -> bool {
        self.is_if_else_block.get()
    }
}
