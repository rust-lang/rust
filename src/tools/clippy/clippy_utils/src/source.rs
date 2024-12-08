//! Utils for extracting, inspecting or transforming source code

#![allow(clippy::module_name_repetitions)]

use rustc_ast::{LitKind, StrStyle};
use rustc_data_structures::sync::Lrc;
use rustc_errors::Applicability;
use rustc_hir::{BlockCheckMode, Expr, ExprKind, UnsafeSource};
use rustc_lint::{EarlyContext, LateContext};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::source_map::{SourceMap, original_sp};
use rustc_span::{
    BytePos, DUMMY_SP, FileNameDisplayPreference, Pos, SourceFile, SourceFileAndLine, Span, SpanData, SyntaxContext,
    hygiene,
};
use std::borrow::Cow;
use std::fmt;
use std::ops::{Deref, Index, Range};

pub trait HasSession {
    fn sess(&self) -> &Session;
}
impl HasSession for Session {
    fn sess(&self) -> &Session {
        self
    }
}
impl HasSession for TyCtxt<'_> {
    fn sess(&self) -> &Session {
        self.sess
    }
}
impl HasSession for EarlyContext<'_> {
    fn sess(&self) -> &Session {
        ::rustc_lint::LintContext::sess(self)
    }
}
impl HasSession for LateContext<'_> {
    fn sess(&self) -> &Session {
        self.tcx.sess()
    }
}

/// Conversion of a value into the range portion of a `Span`.
pub trait SpanRange: Sized {
    fn into_range(self) -> Range<BytePos>;
}
impl SpanRange for Span {
    fn into_range(self) -> Range<BytePos> {
        let data = self.data();
        data.lo..data.hi
    }
}
impl SpanRange for SpanData {
    fn into_range(self) -> Range<BytePos> {
        self.lo..self.hi
    }
}
impl SpanRange for Range<BytePos> {
    fn into_range(self) -> Range<BytePos> {
        self
    }
}

/// Conversion of a value into a `Span`
pub trait IntoSpan: Sized {
    fn into_span(self) -> Span;
    fn with_ctxt(self, ctxt: SyntaxContext) -> Span;
}
impl IntoSpan for Span {
    fn into_span(self) -> Span {
        self
    }
    fn with_ctxt(self, ctxt: SyntaxContext) -> Span {
        self.with_ctxt(ctxt)
    }
}
impl IntoSpan for SpanData {
    fn into_span(self) -> Span {
        self.span()
    }
    fn with_ctxt(self, ctxt: SyntaxContext) -> Span {
        Span::new(self.lo, self.hi, ctxt, self.parent)
    }
}
impl IntoSpan for Range<BytePos> {
    fn into_span(self) -> Span {
        Span::with_root_ctxt(self.start, self.end)
    }
    fn with_ctxt(self, ctxt: SyntaxContext) -> Span {
        Span::new(self.start, self.end, ctxt, None)
    }
}

pub trait SpanRangeExt: SpanRange {
    /// Attempts to get a handle to the source text. Returns `None` if either the span is malformed,
    /// or the source text is not accessible.
    fn get_source_text(self, cx: &impl HasSession) -> Option<SourceText> {
        get_source_range(cx.sess().source_map(), self.into_range()).and_then(SourceText::new)
    }

    /// Gets the source file, and range in the file, of the given span. Returns `None` if the span
    /// extends through multiple files, or is malformed.
    fn get_source_range(self, cx: &impl HasSession) -> Option<SourceFileRange> {
        get_source_range(cx.sess().source_map(), self.into_range())
    }

    /// Calls the given function with the source text referenced and returns the value. Returns
    /// `None` if the source text cannot be retrieved.
    fn with_source_text<T>(self, cx: &impl HasSession, f: impl for<'a> FnOnce(&'a str) -> T) -> Option<T> {
        with_source_text(cx.sess().source_map(), self.into_range(), f)
    }

    /// Checks if the referenced source text satisfies the given predicate. Returns `false` if the
    /// source text cannot be retrieved.
    fn check_source_text(self, cx: &impl HasSession, pred: impl for<'a> FnOnce(&'a str) -> bool) -> bool {
        self.with_source_text(cx, pred).unwrap_or(false)
    }

    /// Calls the given function with the both the text of the source file and the referenced range,
    /// and returns the value. Returns `None` if the source text cannot be retrieved.
    fn with_source_text_and_range<T>(
        self,
        cx: &impl HasSession,
        f: impl for<'a> FnOnce(&'a str, Range<usize>) -> T,
    ) -> Option<T> {
        with_source_text_and_range(cx.sess().source_map(), self.into_range(), f)
    }

    /// Calls the given function with the both the text of the source file and the referenced range,
    /// and creates a new span with the returned range. Returns `None` if the source text cannot be
    /// retrieved, or no result is returned.
    ///
    /// The new range must reside within the same source file.
    fn map_range(
        self,
        cx: &impl HasSession,
        f: impl for<'a> FnOnce(&'a str, Range<usize>) -> Option<Range<usize>>,
    ) -> Option<Range<BytePos>> {
        map_range(cx.sess().source_map(), self.into_range(), f)
    }

    /// Extends the range to include all preceding whitespace characters.
    fn with_leading_whitespace(self, cx: &impl HasSession) -> Range<BytePos> {
        with_leading_whitespace(cx.sess().source_map(), self.into_range())
    }

    /// Trims the leading whitespace from the range.
    fn trim_start(self, cx: &impl HasSession) -> Range<BytePos> {
        trim_start(cx.sess().source_map(), self.into_range())
    }
}
impl<T: SpanRange> SpanRangeExt for T {}

/// Handle to a range of text in a source file.
pub struct SourceText(SourceFileRange);
impl SourceText {
    /// Takes ownership of the source file handle if the source text is accessible.
    pub fn new(text: SourceFileRange) -> Option<Self> {
        if text.as_str().is_some() {
            Some(Self(text))
        } else {
            None
        }
    }

    /// Gets the source text.
    pub fn as_str(&self) -> &str {
        self.0.as_str().unwrap()
    }

    /// Converts this into an owned string.
    pub fn to_owned(&self) -> String {
        self.as_str().to_owned()
    }
}
impl Deref for SourceText {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}
impl AsRef<str> for SourceText {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}
impl<T> Index<T> for SourceText
where
    str: Index<T>,
{
    type Output = <str as Index<T>>::Output;
    fn index(&self, idx: T) -> &Self::Output {
        &self.as_str()[idx]
    }
}
impl fmt::Display for SourceText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

fn get_source_range(sm: &SourceMap, sp: Range<BytePos>) -> Option<SourceFileRange> {
    let start = sm.lookup_byte_offset(sp.start);
    let end = sm.lookup_byte_offset(sp.end);
    if !Lrc::ptr_eq(&start.sf, &end.sf) || start.pos > end.pos {
        return None;
    }
    sm.ensure_source_file_source_present(&start.sf);
    let range = start.pos.to_usize()..end.pos.to_usize();
    Some(SourceFileRange { sf: start.sf, range })
}

fn with_source_text<T>(sm: &SourceMap, sp: Range<BytePos>, f: impl for<'a> FnOnce(&'a str) -> T) -> Option<T> {
    if let Some(src) = get_source_range(sm, sp)
        && let Some(src) = src.as_str()
    {
        Some(f(src))
    } else {
        None
    }
}

fn with_source_text_and_range<T>(
    sm: &SourceMap,
    sp: Range<BytePos>,
    f: impl for<'a> FnOnce(&'a str, Range<usize>) -> T,
) -> Option<T> {
    if let Some(src) = get_source_range(sm, sp)
        && let Some(text) = &src.sf.src
    {
        Some(f(text, src.range))
    } else {
        None
    }
}

#[expect(clippy::cast_possible_truncation)]
fn map_range(
    sm: &SourceMap,
    sp: Range<BytePos>,
    f: impl for<'a> FnOnce(&'a str, Range<usize>) -> Option<Range<usize>>,
) -> Option<Range<BytePos>> {
    if let Some(src) = get_source_range(sm, sp.clone())
        && let Some(text) = &src.sf.src
        && let Some(range) = f(text, src.range.clone())
    {
        debug_assert!(
            range.start <= text.len() && range.end <= text.len(),
            "Range `{range:?}` is outside the source file (file `{}`, length `{}`)",
            src.sf.name.display(FileNameDisplayPreference::Local),
            text.len(),
        );
        debug_assert!(range.start <= range.end, "Range `{range:?}` has overlapping bounds");
        let dstart = (range.start as u32).wrapping_sub(src.range.start as u32);
        let dend = (range.end as u32).wrapping_sub(src.range.start as u32);
        Some(BytePos(sp.start.0.wrapping_add(dstart))..BytePos(sp.start.0.wrapping_add(dend)))
    } else {
        None
    }
}

fn with_leading_whitespace(sm: &SourceMap, sp: Range<BytePos>) -> Range<BytePos> {
    map_range(sm, sp.clone(), |src, range| {
        Some(src.get(..range.start)?.trim_end().len()..range.end)
    })
    .unwrap_or(sp)
}

fn trim_start(sm: &SourceMap, sp: Range<BytePos>) -> Range<BytePos> {
    map_range(sm, sp.clone(), |src, range| {
        let src = src.get(range.clone())?;
        Some(range.start + (src.len() - src.trim_start().len())..range.end)
    })
    .unwrap_or(sp)
}

pub struct SourceFileRange {
    pub sf: Lrc<SourceFile>,
    pub range: Range<usize>,
}
impl SourceFileRange {
    /// Attempts to get the text from the source file. This can fail if the source text isn't
    /// loaded.
    pub fn as_str(&self) -> Option<&str> {
        self.sf
            .src
            .as_ref()
            .map(|src| src.as_str())
            .or_else(|| self.sf.external_src.get().and_then(|src| src.get_source()))
            .and_then(|x| x.get(self.range.clone()))
    }
}

/// Like `snippet_block`, but add braces if the expr is not an `ExprKind::Block`.
pub fn expr_block(
    sess: &impl HasSession,
    expr: &Expr<'_>,
    outer: SyntaxContext,
    default: &str,
    indent_relative_to: Option<Span>,
    app: &mut Applicability,
) -> String {
    let (code, from_macro) = snippet_block_with_context(sess, expr.span, outer, default, indent_relative_to, app);
    if !from_macro
        && let ExprKind::Block(block, _) = expr.kind
        && block.rules != BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided)
    {
        format!("{code}")
    } else {
        // FIXME: add extra indent for the unsafe blocks:
        //     original code:   unsafe { ... }
        //     result code:     { unsafe { ... } }
        //     desired code:    {\n  unsafe { ... }\n}
        format!("{{ {code} }}")
    }
}

/// Returns a new Span that extends the original Span to the first non-whitespace char of the first
/// line.
///
/// ```rust,ignore
///     let x = ();
/// //          ^^
/// // will be converted to
///     let x = ();
/// //  ^^^^^^^^^^
/// ```
pub fn first_line_of_span(sess: &impl HasSession, span: Span) -> Span {
    first_char_in_first_line(sess, span).map_or(span, |first_char_pos| span.with_lo(first_char_pos))
}

fn first_char_in_first_line(sess: &impl HasSession, span: Span) -> Option<BytePos> {
    let line_span = line_span(sess, span);
    snippet_opt(sess, line_span).and_then(|snip| {
        snip.find(|c: char| !c.is_whitespace())
            .map(|pos| line_span.lo() + BytePos::from_usize(pos))
    })
}

/// Extends the span to the beginning of the spans line, incl. whitespaces.
///
/// ```no_run
///        let x = ();
/// //             ^^
/// // will be converted to
///        let x = ();
/// // ^^^^^^^^^^^^^^
/// ```
fn line_span(sess: &impl HasSession, span: Span) -> Span {
    let span = original_sp(span, DUMMY_SP);
    let SourceFileAndLine { sf, line } = sess.sess().source_map().lookup_line(span.lo()).unwrap();
    let line_start = sf.lines()[line];
    let line_start = sf.absolute_position(line_start);
    span.with_lo(line_start)
}

/// Returns the indentation of the line of a span
///
/// ```rust,ignore
/// let x = ();
/// //      ^^ -- will return 0
///     let x = ();
/// //          ^^ -- will return 4
/// ```
pub fn indent_of(sess: &impl HasSession, span: Span) -> Option<usize> {
    snippet_opt(sess, line_span(sess, span)).and_then(|snip| snip.find(|c: char| !c.is_whitespace()))
}

/// Gets a snippet of the indentation of the line of a span
pub fn snippet_indent(sess: &impl HasSession, span: Span) -> Option<String> {
    snippet_opt(sess, line_span(sess, span)).map(|mut s| {
        let len = s.len() - s.trim_start().len();
        s.truncate(len);
        s
    })
}

// If the snippet is empty, it's an attribute that was inserted during macro
// expansion and we want to ignore those, because they could come from external
// sources that the user has no control over.
// For some reason these attributes don't have any expansion info on them, so
// we have to check it this way until there is a better way.
pub fn is_present_in_source(sess: &impl HasSession, span: Span) -> bool {
    if let Some(snippet) = snippet_opt(sess, span) {
        if snippet.is_empty() {
            return false;
        }
    }
    true
}

/// Returns the position just before rarrow
///
/// ```rust,ignore
/// fn into(self) -> () {}
///              ^
/// // in case of unformatted code
/// fn into2(self)-> () {}
///               ^
/// fn into3(self)   -> () {}
///               ^
/// ```
pub fn position_before_rarrow(s: &str) -> Option<usize> {
    s.rfind("->").map(|rpos| {
        let mut rpos = rpos;
        let chars: Vec<char> = s.chars().collect();
        while rpos > 1 {
            if let Some(c) = chars.get(rpos - 1) {
                if c.is_whitespace() {
                    rpos -= 1;
                    continue;
                }
            }
            break;
        }
        rpos
    })
}

/// Reindent a multiline string with possibility of ignoring the first line.
#[expect(clippy::needless_pass_by_value)]
pub fn reindent_multiline(s: Cow<'_, str>, ignore_first: bool, indent: Option<usize>) -> Cow<'_, str> {
    let s_space = reindent_multiline_inner(&s, ignore_first, indent, ' ');
    let s_tab = reindent_multiline_inner(&s_space, ignore_first, indent, '\t');
    reindent_multiline_inner(&s_tab, ignore_first, indent, ' ').into()
}

fn reindent_multiline_inner(s: &str, ignore_first: bool, indent: Option<usize>, ch: char) -> String {
    let x = s
        .lines()
        .skip(usize::from(ignore_first))
        .filter_map(|l| {
            if l.is_empty() {
                None
            } else {
                // ignore empty lines
                Some(l.char_indices().find(|&(_, x)| x != ch).unwrap_or((l.len(), ch)).0)
            }
        })
        .min()
        .unwrap_or(0);
    let indent = indent.unwrap_or(0);
    s.lines()
        .enumerate()
        .map(|(i, l)| {
            if (ignore_first && i == 0) || l.is_empty() {
                l.to_owned()
            } else if x > indent {
                l.split_at(x - indent).1.to_owned()
            } else {
                " ".repeat(indent - x) + l
            }
        })
        .collect::<Vec<String>>()
        .join("\n")
}

/// Converts a span to a code snippet if available, otherwise returns the default.
///
/// This is useful if you want to provide suggestions for your lint or more generally, if you want
/// to convert a given `Span` to a `str`. To create suggestions consider using
/// [`snippet_with_applicability`] to ensure that the applicability stays correct.
///
/// # Example
/// ```rust,ignore
/// // Given two spans one for `value` and one for the `init` expression.
/// let value = Vec::new();
/// //  ^^^^^   ^^^^^^^^^^
/// //  span1   span2
///
/// // The snipped call would return the corresponding code snippet
/// snippet(cx, span1, "..") // -> "value"
/// snippet(cx, span2, "..") // -> "Vec::new()"
/// ```
pub fn snippet<'a>(sess: &impl HasSession, span: Span, default: &'a str) -> Cow<'a, str> {
    snippet_opt(sess, span).map_or_else(|| Cow::Borrowed(default), From::from)
}

/// Same as [`snippet`], but it adapts the applicability level by following rules:
///
/// - Applicability level `Unspecified` will never be changed.
/// - If the span is inside a macro, change the applicability level to `MaybeIncorrect`.
/// - If the default value is used and the applicability level is `MachineApplicable`, change it to
///   `HasPlaceholders`
pub fn snippet_with_applicability<'a>(
    sess: &impl HasSession,
    span: Span,
    default: &'a str,
    applicability: &mut Applicability,
) -> Cow<'a, str> {
    snippet_with_applicability_sess(sess.sess(), span, default, applicability)
}

fn snippet_with_applicability_sess<'a>(
    sess: &Session,
    span: Span,
    default: &'a str,
    applicability: &mut Applicability,
) -> Cow<'a, str> {
    if *applicability != Applicability::Unspecified && span.from_expansion() {
        *applicability = Applicability::MaybeIncorrect;
    }
    snippet_opt(sess, span).map_or_else(
        || {
            if *applicability == Applicability::MachineApplicable {
                *applicability = Applicability::HasPlaceholders;
            }
            Cow::Borrowed(default)
        },
        From::from,
    )
}

/// Converts a span to a code snippet. Returns `None` if not available.
pub fn snippet_opt(sess: &impl HasSession, span: Span) -> Option<String> {
    sess.sess().source_map().span_to_snippet(span).ok()
}

/// Converts a span (from a block) to a code snippet if available, otherwise use default.
///
/// This trims the code of indentation, except for the first line. Use it for blocks or block-like
/// things which need to be printed as such.
///
/// The `indent_relative_to` arg can be used, to provide a span, where the indentation of the
/// resulting snippet of the given span.
///
/// # Example
///
/// ```rust,ignore
/// snippet_block(cx, block.span, "..", None)
/// // where, `block` is the block of the if expr
///     if x {
///         y;
///     }
/// // will return the snippet
/// {
///     y;
/// }
/// ```
///
/// ```rust,ignore
/// snippet_block(cx, block.span, "..", Some(if_expr.span))
/// // where, `block` is the block of the if expr
///     if x {
///         y;
///     }
/// // will return the snippet
/// {
///         y;
///     } // aligned with `if`
/// ```
/// Note that the first line of the snippet always has 0 indentation.
pub fn snippet_block<'a>(
    sess: &impl HasSession,
    span: Span,
    default: &'a str,
    indent_relative_to: Option<Span>,
) -> Cow<'a, str> {
    let snip = snippet(sess, span, default);
    let indent = indent_relative_to.and_then(|s| indent_of(sess, s));
    reindent_multiline(snip, true, indent)
}

/// Same as `snippet_block`, but adapts the applicability level by the rules of
/// `snippet_with_applicability`.
pub fn snippet_block_with_applicability<'a>(
    sess: &impl HasSession,
    span: Span,
    default: &'a str,
    indent_relative_to: Option<Span>,
    applicability: &mut Applicability,
) -> Cow<'a, str> {
    let snip = snippet_with_applicability(sess, span, default, applicability);
    let indent = indent_relative_to.and_then(|s| indent_of(sess, s));
    reindent_multiline(snip, true, indent)
}

pub fn snippet_block_with_context<'a>(
    sess: &impl HasSession,
    span: Span,
    outer: SyntaxContext,
    default: &'a str,
    indent_relative_to: Option<Span>,
    app: &mut Applicability,
) -> (Cow<'a, str>, bool) {
    let (snip, from_macro) = snippet_with_context(sess, span, outer, default, app);
    let indent = indent_relative_to.and_then(|s| indent_of(sess, s));
    (reindent_multiline(snip, true, indent), from_macro)
}

/// Same as `snippet_with_applicability`, but first walks the span up to the given context.
///
/// This will result in the macro call, rather than the expansion, if the span is from a child
/// context. If the span is not from a child context, it will be used directly instead.
///
/// e.g. Given the expression `&vec![]`, getting a snippet from the span for `vec![]` as a HIR node
/// would result in `box []`. If given the context of the address of expression, this function will
/// correctly get a snippet of `vec![]`.
///
/// This will also return whether or not the snippet is a macro call.
pub fn snippet_with_context<'a>(
    sess: &impl HasSession,
    span: Span,
    outer: SyntaxContext,
    default: &'a str,
    applicability: &mut Applicability,
) -> (Cow<'a, str>, bool) {
    snippet_with_context_sess(sess.sess(), span, outer, default, applicability)
}

fn snippet_with_context_sess<'a>(
    sess: &Session,
    span: Span,
    outer: SyntaxContext,
    default: &'a str,
    applicability: &mut Applicability,
) -> (Cow<'a, str>, bool) {
    let (span, is_macro_call) = walk_span_to_context(span, outer).map_or_else(
        || {
            // The span is from a macro argument, and the outer context is the macro using the argument
            if *applicability != Applicability::Unspecified {
                *applicability = Applicability::MaybeIncorrect;
            }
            // TODO: get the argument span.
            (span, false)
        },
        |outer_span| (outer_span, span.ctxt() != outer),
    );

    (
        snippet_with_applicability_sess(sess, span, default, applicability),
        is_macro_call,
    )
}

/// Walks the span up to the target context, thereby returning the macro call site if the span is
/// inside a macro expansion, or the original span if it is not.
///
/// Note this will return `None` in the case of the span being in a macro expansion, but the target
/// context is from expanding a macro argument.
///
/// Given the following
///
/// ```rust,ignore
/// macro_rules! m { ($e:expr) => { f($e) }; }
/// g(m!(0))
/// ```
///
/// If called with a span of the call to `f` and a context of the call to `g` this will return a
/// span containing `m!(0)`. However, if called with a span of the literal `0` this will give a span
/// containing `0` as the context is the same as the outer context.
///
/// This will traverse through multiple macro calls. Given the following:
///
/// ```rust,ignore
/// macro_rules! m { ($e:expr) => { n!($e, 0) }; }
/// macro_rules! n { ($e:expr, $f:expr) => { f($e, $f) }; }
/// g(m!(0))
/// ```
///
/// If called with a span of the call to `f` and a context of the call to `g` this will return a
/// span containing `m!(0)`.
pub fn walk_span_to_context(span: Span, outer: SyntaxContext) -> Option<Span> {
    let outer_span = hygiene::walk_chain(span, outer);
    (outer_span.ctxt() == outer).then_some(outer_span)
}

/// Trims the whitespace from the start and the end of the span.
pub fn trim_span(sm: &SourceMap, span: Span) -> Span {
    let data = span.data();
    let sf: &_ = &sm.lookup_source_file(data.lo);
    let Some(src) = sf.src.as_deref() else {
        return span;
    };
    let Some(snip) = &src.get((data.lo - sf.start_pos).to_usize()..(data.hi - sf.start_pos).to_usize()) else {
        return span;
    };
    let trim_start = snip.len() - snip.trim_start().len();
    let trim_end = snip.len() - snip.trim_end().len();
    SpanData {
        lo: data.lo + BytePos::from_usize(trim_start),
        hi: data.hi - BytePos::from_usize(trim_end),
        ctxt: data.ctxt,
        parent: data.parent,
    }
    .span()
}

/// Expand a span to include a preceding comma
/// ```rust,ignore
/// writeln!(o, "")   ->   writeln!(o, "")
///             ^^                   ^^^^
/// ```
pub fn expand_past_previous_comma(sess: &impl HasSession, span: Span) -> Span {
    let extended = sess.sess().source_map().span_extend_to_prev_char(span, ',', true);
    extended.with_lo(extended.lo() - BytePos(1))
}

/// Converts `expr` to a `char` literal if it's a `str` literal containing a single
/// character (or a single byte with `ascii_only`)
pub fn str_literal_to_char_literal(
    sess: &impl HasSession,
    expr: &Expr<'_>,
    applicability: &mut Applicability,
    ascii_only: bool,
) -> Option<String> {
    if let ExprKind::Lit(lit) = &expr.kind
        && let LitKind::Str(r, style) = lit.node
        && let string = r.as_str()
        && let len = if ascii_only {
            string.len()
        } else {
            string.chars().count()
        }
        && len == 1
    {
        let snip = snippet_with_applicability(sess, expr.span, string, applicability);
        let ch = if let StrStyle::Raw(nhash) = style {
            let nhash = nhash as usize;
            // for raw string: r##"a"##
            &snip[(nhash + 2)..(snip.len() - 1 - nhash)]
        } else {
            // for regular string: "a"
            &snip[1..(snip.len() - 1)]
        };

        let hint = format!("'{}'", match ch {
            "'" => "\\'",
            r"\" => "\\\\",
            "\\\"" => "\"", // no need to escape `"` in `'"'`
            _ => ch,
        });

        Some(hint)
    } else {
        None
    }
}

#[cfg(test)]
mod test {
    use super::reindent_multiline;

    #[test]
    fn test_reindent_multiline_single_line() {
        assert_eq!("", reindent_multiline("".into(), false, None));
        assert_eq!("...", reindent_multiline("...".into(), false, None));
        assert_eq!("...", reindent_multiline("    ...".into(), false, None));
        assert_eq!("...", reindent_multiline("\t...".into(), false, None));
        assert_eq!("...", reindent_multiline("\t\t...".into(), false, None));
    }

    #[test]
    #[rustfmt::skip]
    fn test_reindent_multiline_block() {
        assert_eq!("\
    if x {
        y
    } else {
        z
    }", reindent_multiline("    if x {
            y
        } else {
            z
        }".into(), false, None));
        assert_eq!("\
    if x {
    \ty
    } else {
    \tz
    }", reindent_multiline("    if x {
        \ty
        } else {
        \tz
        }".into(), false, None));
    }

    #[test]
    #[rustfmt::skip]
    fn test_reindent_multiline_empty_line() {
        assert_eq!("\
    if x {
        y

    } else {
        z
    }", reindent_multiline("    if x {
            y

        } else {
            z
        }".into(), false, None));
    }

    #[test]
    #[rustfmt::skip]
    fn test_reindent_multiline_lines_deeper() {
        assert_eq!("\
        if x {
            y
        } else {
            z
        }", reindent_multiline("\
    if x {
        y
    } else {
        z
    }".into(), true, Some(8)));
    }
}
