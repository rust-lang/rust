use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{SpanRangeExt, snippet_indent};
use clippy_utils::tokenize_with_text;
use itertools::Itertools;
use rustc_ast::AttrStyle;
use rustc_ast::token::CommentKind;
use rustc_errors::{Applicability, Diag, SuggestionStyle};
use rustc_hir::{AttrKind, Attribute, ItemKind, Node};
use rustc_lexer::TokenKind;
use rustc_lint::LateContext;
use rustc_span::{BytePos, ExpnKind, InnerSpan, Span, SpanData};

use super::{EMPTY_LINE_AFTER_DOC_COMMENTS, EMPTY_LINE_AFTER_OUTER_ATTR};

#[derive(Debug, PartialEq, Clone, Copy)]
enum StopKind {
    Attr,
    Doc(CommentKind),
}

impl StopKind {
    fn is_doc(self) -> bool {
        matches!(self, StopKind::Doc(_))
    }
}

#[derive(Debug)]
struct Stop {
    span: Span,
    kind: StopKind,
    first: usize,
    last: usize,
}

impl Stop {
    fn convert_to_inner(&self) -> (Span, String) {
        let inner = match self.kind {
            // #|[...]
            StopKind::Attr => InnerSpan::new(1, 1),
            // /// or /**
            //   ^      ^
            StopKind::Doc(_) => InnerSpan::new(2, 3),
        };
        (self.span.from_inner(inner), "!".into())
    }

    fn comment_out(&self, cx: &LateContext<'_>, suggestions: &mut Vec<(Span, String)>) {
        match self.kind {
            StopKind::Attr => {
                if cx.tcx.sess.source_map().is_multiline(self.span) {
                    suggestions.extend([
                        (self.span.shrink_to_lo(), "/* ".into()),
                        (self.span.shrink_to_hi(), " */".into()),
                    ]);
                } else {
                    suggestions.push((self.span.shrink_to_lo(), "// ".into()));
                }
            },
            StopKind::Doc(CommentKind::Line) => suggestions.push((self.span.shrink_to_lo(), "// ".into())),
            StopKind::Doc(CommentKind::Block) => {
                // /** outer */  /*! inner */
                //  ^             ^
                let asterisk = self.span.from_inner(InnerSpan::new(1, 2));
                suggestions.push((asterisk, String::new()));
            },
        }
    }

    fn from_attr(cx: &LateContext<'_>, attr: &Attribute) -> Option<Self> {
        let SpanData { lo, hi, .. } = attr.span.data();
        let file = cx.tcx.sess.source_map().lookup_source_file(lo);

        Some(Self {
            span: attr.span,
            kind: match attr.kind {
                AttrKind::Normal(_) => StopKind::Attr,
                AttrKind::DocComment(comment_kind, _) => StopKind::Doc(comment_kind),
            },
            first: file.lookup_line(file.relative_position(lo))?,
            last: file.lookup_line(file.relative_position(hi))?,
        })
    }
}

/// Represents a set of attrs/doc comments separated by 1 or more empty lines
///
/// ```ignore
/// /// chunk 1 docs
/// // not an empty line so also part of chunk 1
/// #[chunk_1_attrs] // <-- prev_stop
///
/// /* gap */
///
/// /// chunk 2 docs // <-- next_stop
/// #[chunk_2_attrs]
/// ```
struct Gap<'a> {
    /// The span of individual empty lines including the newline at the end of the line
    empty_lines: Vec<Span>,
    has_comment: bool,
    next_stop: &'a Stop,
    prev_stop: &'a Stop,
    /// The chunk that includes [`prev_stop`](Self::prev_stop)
    prev_chunk: &'a [Stop],
}

impl<'a> Gap<'a> {
    fn new(cx: &LateContext<'_>, prev_chunk: &'a [Stop], next_chunk: &'a [Stop]) -> Option<Self> {
        let prev_stop = prev_chunk.last()?;
        let next_stop = next_chunk.first()?;
        let gap_span = prev_stop.span.between(next_stop.span);
        let gap_snippet = gap_span.get_source_text(cx)?;

        let mut has_comment = false;
        let mut empty_lines = Vec::new();

        for (token, source, inner_span) in tokenize_with_text(&gap_snippet) {
            match token {
                TokenKind::BlockComment {
                    doc_style: None,
                    terminated: true,
                }
                | TokenKind::LineComment { doc_style: None } => has_comment = true,
                TokenKind::Whitespace => {
                    let newlines = source.bytes().positions(|b| b == b'\n');
                    empty_lines.extend(
                        newlines
                            .tuple_windows()
                            .map(|(a, b)| InnerSpan::new(inner_span.start + a + 1, inner_span.start + b))
                            .map(|inner_span| gap_span.from_inner(inner_span)),
                    );
                },
                // Ignore cfg_attr'd out attributes as they may contain empty lines, could also be from macro
                // shenanigans
                _ => return None,
            }
        }

        (!empty_lines.is_empty()).then_some(Self {
            empty_lines,
            has_comment,
            next_stop,
            prev_stop,
            prev_chunk,
        })
    }

    fn contiguous_empty_lines(&self) -> impl Iterator<Item = Span> + '_ {
        self.empty_lines
            // The `+ BytePos(1)` means "next line", because each empty line span is "N:1-N:1".
            .chunk_by(|a, b| a.hi() + BytePos(1) == b.lo())
            .map(|chunk| {
                let first = chunk.first().expect("at least one empty line");
                let last = chunk.last().expect("at least one empty line");
                // The BytePos subtraction here is safe, as before an empty line, there must be at least one
                // attribute/comment. The span needs to start at the end of the previous line.
                first.with_lo(first.lo() - BytePos(1)).with_hi(last.hi())
            })
    }
}

/// If the node the attributes/docs apply to is the first in the module/crate suggest converting
/// them to inner attributes/docs
fn suggest_inner(cx: &LateContext<'_>, diag: &mut Diag<'_, ()>, kind: StopKind, gaps: &[Gap<'_>]) {
    let Some(owner) = cx.last_node_with_lint_attrs.as_owner() else {
        return;
    };
    let parent_desc = match cx.tcx.parent_hir_node(owner.into()) {
        Node::Item(item)
            if let ItemKind::Mod(parent_mod) = item.kind
                && let [first, ..] = parent_mod.item_ids
                && first.owner_id == owner =>
        {
            "parent module"
        },
        Node::Crate(crate_mod)
            if let Some(first) = crate_mod
                .item_ids
                .iter()
                .map(|&id| cx.tcx.hir().item(id))
                // skip prelude imports
                .find(|item| !matches!(item.span.ctxt().outer_expn_data().kind, ExpnKind::AstPass(_)))
                && first.owner_id == owner =>
        {
            "crate"
        },
        _ => return,
    };

    diag.multipart_suggestion_verbose(
        match kind {
            StopKind::Attr => format!("if the attribute should apply to the {parent_desc} use an inner attribute"),
            StopKind::Doc(_) => format!("if the comment should document the {parent_desc} use an inner doc comment"),
        },
        gaps.iter()
            .flat_map(|gap| gap.prev_chunk)
            .map(Stop::convert_to_inner)
            .collect(),
        Applicability::MaybeIncorrect,
    );
}

fn check_gaps(cx: &LateContext<'_>, gaps: &[Gap<'_>]) -> bool {
    let Some(first_gap) = gaps.first() else {
        return false;
    };
    let empty_lines = || gaps.iter().flat_map(|gap| gap.empty_lines.iter().copied());
    let contiguous_empty_lines = || gaps.iter().flat_map(Gap::contiguous_empty_lines);
    let mut has_comment = false;
    let mut has_attr = false;
    for gap in gaps {
        has_comment |= gap.has_comment;
        if !has_attr {
            has_attr = gap.prev_chunk.iter().any(|stop| stop.kind == StopKind::Attr);
        }
    }
    let kind = first_gap.prev_stop.kind;
    let (lint, kind_desc) = match kind {
        StopKind::Attr => (EMPTY_LINE_AFTER_OUTER_ATTR, "outer attribute"),
        StopKind::Doc(_) => (EMPTY_LINE_AFTER_DOC_COMMENTS, "doc comment"),
    };
    let (lines, are, them) = if empty_lines().nth(1).is_some() {
        ("lines", "are", "them")
    } else {
        ("line", "is", "it")
    };
    span_lint_and_then(
        cx,
        lint,
        first_gap.prev_stop.span.to(empty_lines().last().unwrap()),
        format!("empty {lines} after {kind_desc}"),
        |diag| {
            if let Some(owner) = cx.last_node_with_lint_attrs.as_owner() {
                let def_id = owner.to_def_id();
                let def_descr = cx.tcx.def_descr(def_id);
                diag.span_label(cx.tcx.def_span(def_id), match kind {
                    StopKind::Attr => format!("the attribute applies to this {def_descr}"),
                    StopKind::Doc(_) => format!("the comment documents this {def_descr}"),
                });
            }

            diag.multipart_suggestion_with_style(
                format!("if the empty {lines} {are} unintentional remove {them}"),
                contiguous_empty_lines()
                    .map(|empty_lines| (empty_lines, String::new()))
                    .collect(),
                Applicability::MaybeIncorrect,
                SuggestionStyle::HideCodeAlways,
            );

            if has_comment && kind.is_doc() {
                // Likely doc comments that applied to some now commented out code
                //
                // /// Old docs for Foo
                // // struct Foo;

                let mut suggestions = Vec::new();
                for stop in gaps.iter().flat_map(|gap| gap.prev_chunk) {
                    stop.comment_out(cx, &mut suggestions);
                }
                let name = match cx.tcx.hir().opt_name(cx.last_node_with_lint_attrs) {
                    Some(name) => format!("`{name}`"),
                    None => "this".into(),
                };
                diag.multipart_suggestion_verbose(
                    format!("if the doc comment should not document {name} comment it out"),
                    suggestions,
                    Applicability::MaybeIncorrect,
                );
            } else {
                suggest_inner(cx, diag, kind, gaps);
            }

            if kind == StopKind::Doc(CommentKind::Line)
                && gaps
                    .iter()
                    .all(|gap| !gap.has_comment && gap.next_stop.kind == StopKind::Doc(CommentKind::Line))
            {
                // Commentless empty gaps between line doc comments, possibly intended to be part of the markdown

                let indent = snippet_indent(cx, first_gap.prev_stop.span).unwrap_or_default();
                diag.multipart_suggestion_verbose(
                    format!("if the documentation should include the empty {lines} include {them} in the comment"),
                    empty_lines()
                        .map(|empty_line| (empty_line, format!("{indent}///")))
                        .collect(),
                    Applicability::MaybeIncorrect,
                );
            }
        },
    );
    kind.is_doc()
}

/// Returns `true` if [`EMPTY_LINE_AFTER_DOC_COMMENTS`] triggered, used to skip other doc comment
/// lints where they would be confusing
///
/// [`EMPTY_LINE_AFTER_OUTER_ATTR`] is also here to share an implementation but does not return
/// `true` if it triggers
pub(super) fn check(cx: &LateContext<'_>, attrs: &[Attribute]) -> bool {
    let mut outer = attrs
        .iter()
        .filter(|attr| attr.style == AttrStyle::Outer && !attr.span.from_expansion())
        .map(|attr| Stop::from_attr(cx, attr))
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();

    if outer.is_empty() {
        return false;
    }

    // Push a fake attribute Stop for the item itself so we check for gaps between the last outer
    // attr/doc comment and the item they apply to
    let span = cx.tcx.hir().span(cx.last_node_with_lint_attrs);
    if !span.from_expansion()
        && let Ok(line) = cx.tcx.sess.source_map().lookup_line(span.lo())
    {
        outer.push(Stop {
            span,
            kind: StopKind::Attr,
            first: line.line,
            // last doesn't need to be accurate here, we don't compare it with anything
            last: line.line,
        });
    }

    let mut gaps = Vec::new();
    let mut last = 0;
    for pos in outer
        .array_windows()
        .positions(|[a, b]| b.first.saturating_sub(a.last) > 1)
    {
        // we want to be after the first stop in the window
        let pos = pos + 1;
        if let Some(gap) = Gap::new(cx, &outer[last..pos], &outer[pos..]) {
            last = pos;
            gaps.push(gap);
        }
    }

    check_gaps(cx, &gaps)
}
