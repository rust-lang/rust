// A generic trait to abstract the rewriting of an element (of the AST).

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use rustc_span::Span;
use thiserror::Error;

use crate::FormatReport;
use crate::config::{Config, IndentStyle};
use crate::parse::session::ParseSess;
use crate::shape::Shape;
use crate::skip::SkipContext;
use crate::visitor::SnippetProvider;

pub(crate) type RewriteResult = Result<String, RewriteError>;
pub(crate) trait Rewrite {
    /// Rewrite self into shape.
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String>;

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        self.rewrite(context, shape).unknown_error()
    }
}

impl<T: Rewrite> Rewrite for Box<T> {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        (**self).rewrite(context, shape)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MacroErrorKind {
    ParseFailure,
    ReplaceMacroVariable,
    Unknown,
}

impl std::fmt::Display for MacroErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MacroErrorKind::ParseFailure => write!(f, "(parse failure)"),
            MacroErrorKind::ReplaceMacroVariable => write!(f, "(replacing macro variables with $)"),
            MacroErrorKind::Unknown => write!(f, ""),
        }
    }
}

#[derive(Clone, Error, Debug)]
pub(crate) enum RewriteError {
    #[error("Formatting was skipped due to skip attribute or out of file range.")]
    SkipFormatting,

    #[error("It exceeds the required width of {configured_width} for the span: {span:?}")]
    ExceedsMaxWidth { configured_width: usize, span: Span },

    #[error("Failed to format given macro{} at: {span:?}", kind)]
    MacroFailure { kind: MacroErrorKind, span: Span },

    /// Format failure that does not fit to above categories.
    #[error("An unknown error occurred during formatting.")]
    Unknown,
}

/// Extension trait used to conveniently convert to RewriteError
pub(crate) trait RewriteErrorExt<T> {
    fn max_width_error(self, width: usize, span: Span) -> Result<T, RewriteError>;
    fn macro_error(self, kind: MacroErrorKind, span: Span) -> Result<T, RewriteError>;
    fn unknown_error(self) -> Result<T, RewriteError>;
}

impl<T> RewriteErrorExt<T> for Option<T> {
    fn max_width_error(self, width: usize, span: Span) -> Result<T, RewriteError> {
        self.ok_or_else(|| RewriteError::ExceedsMaxWidth {
            configured_width: width,
            span: span,
        })
    }

    fn macro_error(self, kind: MacroErrorKind, span: Span) -> Result<T, RewriteError> {
        self.ok_or_else(|| RewriteError::MacroFailure {
            kind: kind,
            span: span,
        })
    }

    fn unknown_error(self) -> Result<T, RewriteError> {
        self.ok_or_else(|| RewriteError::Unknown)
    }
}

#[derive(Clone)]
pub(crate) struct RewriteContext<'a> {
    pub(crate) psess: &'a ParseSess,
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
