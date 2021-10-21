use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg};
use clippy_utils::source::{indent_of, reindent_multiline, snippet};
use clippy_utils::{in_macro, is_lint_allowed};
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{Block, BlockCheckMode, Expr, ExprKind, HirId, Local, UnsafeSource};
use rustc_lexer::TokenKind;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::TyCtxt;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{BytePos, Span};
use std::borrow::Cow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `unsafe` blocks without a `// Safety: ` comment
    /// explaining why the unsafe operations performed inside
    /// the block are safe.
    ///
    /// ### Why is this bad?
    /// Undocumented unsafe blocks can make it difficult to
    /// read and maintain code, as well as uncover unsoundness
    /// and bugs.
    ///
    /// ### Example
    /// ```rust
    /// use std::ptr::NonNull;
    /// let a = &mut 42;
    ///
    /// let ptr = unsafe { NonNull::new_unchecked(a) };
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::ptr::NonNull;
    /// let a = &mut 42;
    ///
    /// // Safety: references are guaranteed to be non-null.
    /// let ptr = unsafe { NonNull::new_unchecked(a) };
    /// ```
    pub UNDOCUMENTED_UNSAFE_BLOCKS,
    restriction,
    "creating an unsafe block without explaining why it is safe"
}

impl_lint_pass!(UndocumentedUnsafeBlocks => [UNDOCUMENTED_UNSAFE_BLOCKS]);

#[derive(Default)]
pub struct UndocumentedUnsafeBlocks {
    pub local_level: u32,
    pub local_span: Option<Span>,
    // The local was already checked for an overall safety comment
    // There is no need to continue checking the blocks in the local
    pub local_checked: bool,
    // Since we can only check the blocks from expanded macros
    // We have to omit the suggestion due to the actual definition
    // Not being available to us
    pub macro_expansion: bool,
}

impl LateLintPass<'_> for UndocumentedUnsafeBlocks {
    fn check_block(&mut self, cx: &LateContext<'_>, block: &'_ Block<'_>) {
        if_chain! {
            if !self.local_checked;
            if !is_lint_allowed(cx, UNDOCUMENTED_UNSAFE_BLOCKS, block.hir_id);
            if !in_external_macro(cx.tcx.sess, block.span);
            if let BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided) = block.rules;
            if let Some(enclosing_scope_hir_id) = cx.tcx.hir().get_enclosing_scope(block.hir_id);
            if self.block_has_safety_comment(cx.tcx, enclosing_scope_hir_id, block.span) == Some(false);
            then {
                let mut span = block.span;

                if let Some(local_span) = self.local_span {
                    span = local_span;

                    let result = self.block_has_safety_comment(cx.tcx, enclosing_scope_hir_id, span);

                    if result.unwrap_or(true) {
                        self.local_checked = true;
                        return;
                    }
                }

                self.lint(cx, span);
            }
        }
    }

    fn check_local(&mut self, cx: &LateContext<'_>, local: &'_ Local<'_>) {
        if_chain! {
            if !is_lint_allowed(cx, UNDOCUMENTED_UNSAFE_BLOCKS, local.hir_id);
            if !in_external_macro(cx.tcx.sess, local.span);
            if let Some(init) = local.init;
            then {
                self.visit_expr(init);

                if self.local_level > 0 {
                    self.local_span = Some(local.span);
                }
            }
        }
    }

    fn check_block_post(&mut self, _: &LateContext<'_>, _: &'_ Block<'_>) {
        self.local_level = self.local_level.saturating_sub(1);

        if self.local_level == 0 {
            self.local_checked = false;
            self.local_span = None;
        }
    }
}

impl<'hir> Visitor<'hir> for UndocumentedUnsafeBlocks {
    type Map = Map<'hir>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, ex: &'v Expr<'v>) {
        match ex.kind {
            ExprKind::Block(_, _) => self.local_level = self.local_level.saturating_add(1),
            _ => walk_expr(self, ex),
        }
    }
}

impl UndocumentedUnsafeBlocks {
    fn block_has_safety_comment(&mut self, tcx: TyCtxt<'_>, enclosing_hir_id: HirId, block_span: Span) -> Option<bool> {
        let map = tcx.hir();
        let source_map = tcx.sess.source_map();

        let enclosing_scope_span = map.opt_span(enclosing_hir_id)?;

        let between_span = if in_macro(block_span) {
            self.macro_expansion = true;
            enclosing_scope_span.with_hi(block_span.hi())
        } else {
            self.macro_expansion = false;
            enclosing_scope_span.to(block_span)
        };

        let file_name = source_map.span_to_filename(between_span);
        let source_file = source_map.get_source_file(&file_name)?;

        let lex_start = (between_span.lo().0 + 1) as usize;
        let src_str = source_file.src.as_ref()?[lex_start..between_span.hi().0 as usize].to_string();

        let mut pos = 0;
        let mut comment = false;

        for token in rustc_lexer::tokenize(&src_str) {
            match token.kind {
                TokenKind::LineComment { doc_style: None }
                | TokenKind::BlockComment {
                    doc_style: None,
                    terminated: true,
                } => {
                    let comment_str = src_str[pos + 2..pos + token.len].to_ascii_uppercase();

                    if comment_str.contains("SAFETY:") {
                        comment = true;
                    }
                },
                // We need to add all whitespace to `pos` before checking the comment's line number
                TokenKind::Whitespace => {},
                _ => {
                    if comment {
                        // Get the line number of the "comment" (really wherever the trailing whitespace ended)
                        let comment_line_num = source_file
                            .lookup_file_pos_with_col_display(BytePos((lex_start + pos).try_into().unwrap()))
                            .0;
                        // Find the block/local's line number
                        let block_line_num = tcx.sess.source_map().lookup_char_pos(block_span.lo()).line;

                        // Check the comment is immediately followed by the block/local
                        if block_line_num == comment_line_num + 1 || block_line_num == comment_line_num {
                            return Some(true);
                        }

                        comment = false;
                    }
                },
            }

            pos += token.len;
        }

        Some(false)
    }

    fn lint(&self, cx: &LateContext<'_>, mut span: Span) {
        let source_map = cx.tcx.sess.source_map();

        if source_map.is_multiline(span) {
            span = source_map.span_until_char(span, '\n');
        }

        if self.macro_expansion {
            span_lint_and_help(
                cx,
                UNDOCUMENTED_UNSAFE_BLOCKS,
                span,
                "unsafe block in macro expansion missing a safety comment",
                None,
                "consider adding a safety comment in the macro definition",
            );
        } else {
            let block_indent = indent_of(cx, span);
            let suggestion = format!("// Safety: ...\n{}", snippet(cx, span, ".."));

            span_lint_and_sugg(
                cx,
                UNDOCUMENTED_UNSAFE_BLOCKS,
                span,
                "unsafe block missing a safety comment",
                "consider adding a safety comment",
                reindent_multiline(Cow::Borrowed(&suggestion), true, block_indent).to_string(),
                Applicability::HasPlaceholders,
            );
        }
    }
}
