use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_lint_allowed;
use clippy_utils::source::walk_span_to_context;
use rustc_data_structures::sync::Lrc;
use rustc_hir::{Block, BlockCheckMode, UnsafeSource};
use rustc_lexer::{tokenize, TokenKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{BytePos, Pos, SyntaxContext};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `unsafe` blocks without a `// SAFETY: ` comment
    /// explaining why the unsafe operations performed inside
    /// the block are safe.
    ///
    /// Note the comment must appear on the line(s) preceding the unsafe block
    /// with nothing appearing in between. The following is ok:
    /// ```ignore
    /// foo(
    ///     // SAFETY:
    ///     // This is a valid safety comment
    ///     unsafe { *x }
    /// )
    /// ```
    /// But neither of these are:
    /// ```ignore
    /// // SAFETY:
    /// // This is not a valid safety comment
    /// foo(
    ///     /* SAFETY: Neither is this */ unsafe { *x },
    /// );
    /// ```
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
    /// // SAFETY: references are guaranteed to be non-null.
    /// let ptr = unsafe { NonNull::new_unchecked(a) };
    /// ```
    #[clippy::version = "1.58.0"]
    pub UNDOCUMENTED_UNSAFE_BLOCKS,
    restriction,
    "creating an unsafe block without explaining why it is safe"
}

declare_lint_pass!(UndocumentedUnsafeBlocks => [UNDOCUMENTED_UNSAFE_BLOCKS]);

impl LateLintPass<'_> for UndocumentedUnsafeBlocks {
    fn check_block(&mut self, cx: &LateContext<'_>, block: &'_ Block<'_>) {
        if block.rules == BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided)
            && !in_external_macro(cx.tcx.sess, block.span)
            && !is_lint_allowed(cx, UNDOCUMENTED_UNSAFE_BLOCKS, block.hir_id)
            && !is_unsafe_from_proc_macro(cx, block)
            && !block_has_safety_comment(cx, block)
        {
            let source_map = cx.tcx.sess.source_map();
            let span = if source_map.is_multiline(block.span) {
                source_map.span_until_char(block.span, '\n')
            } else {
                block.span
            };

            span_lint_and_help(
                cx,
                UNDOCUMENTED_UNSAFE_BLOCKS,
                span,
                "unsafe block missing a safety comment",
                None,
                "consider adding a safety comment on the preceding line",
            );
        }
    }
}

fn is_unsafe_from_proc_macro(cx: &LateContext<'_>, block: &Block<'_>) -> bool {
    let source_map = cx.sess().source_map();
    let file_pos = source_map.lookup_byte_offset(block.span.lo());
    file_pos
        .sf
        .src
        .as_deref()
        .and_then(|src| src.get(file_pos.pos.to_usize()..))
        .map_or(true, |src| !src.starts_with("unsafe"))
}

/// Checks if the lines immediately preceding the block contain a safety comment.
fn block_has_safety_comment(cx: &LateContext<'_>, block: &Block<'_>) -> bool {
    // This intentionally ignores text before the start of a function so something like:
    // ```
    //     // SAFETY: reason
    //     fn foo() { unsafe { .. } }
    // ```
    // won't work. This is to avoid dealing with where such a comment should be place relative to
    // attributes and doc comments.

    let source_map = cx.sess().source_map();
    let ctxt = block.span.ctxt();
    if ctxt != SyntaxContext::root() {
        // From a macro expansion. Get the text from the start of the macro declaration to start of the unsafe block.
        //     macro_rules! foo { () => { stuff }; (x) => { unsafe { stuff } }; }
        //     ^--------------------------------------------^
        if let Ok(unsafe_line) = source_map.lookup_line(block.span.lo())
            && let Ok(macro_line) = source_map.lookup_line(ctxt.outer_expn_data().def_site.lo())
            && Lrc::ptr_eq(&unsafe_line.sf, &macro_line.sf)
            && let Some(src) = unsafe_line.sf.src.as_deref()
        {
            macro_line.line < unsafe_line.line && text_has_safety_comment(
                src,
                &unsafe_line.sf.lines[macro_line.line + 1..=unsafe_line.line],
                unsafe_line.sf.start_pos.to_usize(),
            )
        } else {
            // Problem getting source text. Pretend a comment was found.
            true
        }
    } else if let Ok(unsafe_line) = source_map.lookup_line(block.span.lo())
        && let Some(body) = cx.enclosing_body
        && let Some(body_span) = walk_span_to_context(cx.tcx.hir().body(body).value.span, SyntaxContext::root())
        && let Ok(body_line) = source_map.lookup_line(body_span.lo())
        && Lrc::ptr_eq(&unsafe_line.sf, &body_line.sf)
        && let Some(src) = unsafe_line.sf.src.as_deref()
    {
        // Get the text from the start of function body to the unsafe block.
        //     fn foo() { some_stuff; unsafe { stuff }; other_stuff; }
        //              ^-------------^
        body_line.line < unsafe_line.line && text_has_safety_comment(
            src,
            &unsafe_line.sf.lines[body_line.line + 1..=unsafe_line.line],
            unsafe_line.sf.start_pos.to_usize(),
        )
    } else {
        // Problem getting source text. Pretend a comment was found.
        true
    }
}

/// Checks if the given text has a safety comment for the immediately proceeding line.
fn text_has_safety_comment(src: &str, line_starts: &[BytePos], offset: usize) -> bool {
    let mut lines = line_starts
        .array_windows::<2>()
        .rev()
        .map_while(|[start, end]| {
            let start = start.to_usize() - offset;
            let end = end.to_usize() - offset;
            src.get(start..end).map(|text| (start, text.trim_start()))
        })
        .filter(|(_, text)| !text.is_empty());

    let Some((line_start, line)) = lines.next() else {
        return false;
    };
    // Check for a sequence of line comments.
    if line.starts_with("//") {
        let mut line = line;
        loop {
            if line.to_ascii_uppercase().contains("SAFETY:") {
                return true;
            }
            match lines.next() {
                Some((_, x)) if x.starts_with("//") => line = x,
                _ => return false,
            }
        }
    }
    // No line comments; look for the start of a block comment.
    // This will only find them if they are at the start of a line.
    let (mut line_start, mut line) = (line_start, line);
    loop {
        if line.starts_with("/*") {
            let src = src[line_start..line_starts.last().unwrap().to_usize() - offset].trim_start();
            let mut tokens = tokenize(src);
            return src[..tokens.next().unwrap().len]
                .to_ascii_uppercase()
                .contains("SAFETY:")
                && tokens.all(|t| t.kind == TokenKind::Whitespace);
        }
        match lines.next() {
            Some(x) => (line_start, line) = x,
            None => return false,
        }
    }
}
