use rustc_hir::def_id::DefId;
use rustc_middle::hir;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::MirSpanview;
use rustc_span::{BytePos, Pos, Span};

use std::io::{self, Write};
use std::iter::Peekable;

pub const TOOLTIP_INDENT: &str = "    ";

const NEW_LINE_SPAN: &str = "</span>\n<span class=\"line\">";
const HEADER: &str = r#"<!DOCTYPE html>
<html>
<head>
    <title>coverage_of_if_else - Code Regions</title>
    <style>
    .line {
        counter-increment: line;
    }
    .line:before {
        content: counter(line) ": ";
        font-family: Menlo, Monaco, monospace;
        font-style: italic;
        width: 3.8em;
        display: inline-block;
        text-align: right;
        filter: opacity(50%);
        -webkit-user-select: none;
    }
    .code {
        color: #dddddd;
        background-color: #222222;
        font-family: Menlo, Monaco, monospace;
        line-height: 1.4em;
        border-bottom: 2px solid #222222;
        white-space: pre;
        display: inline-block;
    }
    .odd {
        background-color: #55bbff;
        color: #223311;
    }
    .even {
        background-color: #ee7756;
        color: #551133;
    }
    .code {
        --index: calc(var(--layer) - 1);
        padding-top: calc(var(--index) * 0.15em);
        filter:
            hue-rotate(calc(var(--index) * 25deg))
            saturate(calc(100% - (var(--index) * 2%)))
            brightness(calc(100% - (var(--index) * 1.5%)));
    }
    .annotation {
        color: #4444ff;
        font-family: monospace;
        font-style: italic;
        display: none;
        -webkit-user-select: none;
    }
    body:active .annotation {
        /* requires holding mouse down anywhere on the page */
        display: inline-block;
    }
    span:hover .annotation {
        /* requires hover over a span ONLY on its first line */
        display: inline-block;
    }
    </style>
</head>
<body>"#;

const FOOTER: &str = r#"
</body>
</html>"#;

/// Metadata to highlight the span of a MIR BasicBlock, Statement, or Terminator.
pub struct SpanViewable {
    pub span: Span,
    pub title: String,
    pub tooltip: String,
}

/// Write a spanview HTML+CSS file to analyze MIR element spans.
pub fn write_mir_fn_spanview<'tcx, W>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    body: &Body<'tcx>,
    spanview: MirSpanview,
    w: &mut W,
) -> io::Result<()>
where
    W: Write,
{
    let body_span = hir_body(tcx, def_id).value.span;
    let mut span_viewables = Vec::new();
    for (bb, data) in body.basic_blocks().iter_enumerated() {
        match spanview {
            MirSpanview::Statement => {
                for (i, statement) in data.statements.iter().enumerate() {
                    if let Some(span_viewable) =
                        statement_span_viewable(tcx, body_span, bb, i, statement)
                    {
                        span_viewables.push(span_viewable);
                    }
                }
                if let Some(span_viewable) = terminator_span_viewable(tcx, body_span, bb, data) {
                    span_viewables.push(span_viewable);
                }
            }
            MirSpanview::Terminator => {
                if let Some(span_viewable) = terminator_span_viewable(tcx, body_span, bb, data) {
                    span_viewables.push(span_viewable);
                }
            }
            MirSpanview::Block => {
                if let Some(span_viewable) = block_span_viewable(tcx, body_span, bb, data) {
                    span_viewables.push(span_viewable);
                }
            }
        }
    }
    write_spanview_document(tcx, def_id, span_viewables, w)?;
    Ok(())
}

/// Generate a spanview HTML+CSS document for the given local function `def_id`, and a pre-generated
/// list `SpanViewable`s.
pub fn write_spanview_document<'tcx, W>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    mut span_viewables: Vec<SpanViewable>,
    w: &mut W,
) -> io::Result<()>
where
    W: Write,
{
    let fn_span = fn_span(tcx, def_id);
    writeln!(w, "{}", HEADER)?;
    let mut next_pos = fn_span.lo();
    let end_pos = fn_span.hi();
    let source_map = tcx.sess.source_map();
    let start = source_map.lookup_char_pos(next_pos);
    write!(
        w,
        r#"<div class="code" style="counter-reset: line {}"><span class="line">{}"#,
        start.line - 1,
        " ".repeat(start.col.to_usize())
    )?;
    span_viewables.sort_unstable_by(|a, b| {
        let a = a.span;
        let b = b.span;
        if a.lo() == b.lo() {
            // Sort hi() in reverse order so shorter spans are attempted after longer spans.
            // This should give shorter spans a higher "layer", so they are not covered by
            // the longer spans.
            b.hi().partial_cmp(&a.hi())
        } else {
            a.lo().partial_cmp(&b.lo())
        }
        .unwrap()
    });
    let mut ordered_span_viewables = span_viewables.iter().peekable();
    let mut alt = false;
    while ordered_span_viewables.peek().is_some() {
        next_pos = write_span_viewables(tcx, next_pos, &mut ordered_span_viewables, false, 1, w)?;
        alt = !alt;
    }
    if next_pos < end_pos {
        write_coverage_gap(tcx, next_pos, end_pos, w)?;
    }
    write!(w, r#"</span></div>"#)?;
    writeln!(w, "{}", FOOTER)?;
    Ok(())
}

/// Format a string showing the start line and column, and end line and column within a file.
pub fn source_range_no_file<'tcx>(tcx: TyCtxt<'tcx>, span: &Span) -> String {
    let source_map = tcx.sess.source_map();
    let start = source_map.lookup_char_pos(span.lo());
    let end = source_map.lookup_char_pos(span.hi());
    format!("{}:{}-{}:{}", start.line, start.col.to_usize() + 1, end.line, end.col.to_usize() + 1)
}

pub fn statement_kind_name(statement: &Statement<'_>) -> &'static str {
    use StatementKind::*;
    match statement.kind {
        Assign(..) => "Assign",
        FakeRead(..) => "FakeRead",
        SetDiscriminant { .. } => "SetDiscriminant",
        StorageLive(..) => "StorageLive",
        StorageDead(..) => "StorageDead",
        LlvmInlineAsm(..) => "LlvmInlineAsm",
        Retag(..) => "Retag",
        AscribeUserType(..) => "AscribeUserType",
        Coverage(..) => "Coverage",
        Nop => "Nop",
    }
}

pub fn terminator_kind_name(term: &Terminator<'_>) -> &'static str {
    use TerminatorKind::*;
    match term.kind {
        Goto { .. } => "Goto",
        SwitchInt { .. } => "SwitchInt",
        Resume => "Resume",
        Abort => "Abort",
        Return => "Return",
        Unreachable => "Unreachable",
        Drop { .. } => "Drop",
        DropAndReplace { .. } => "DropAndReplace",
        Call { .. } => "Call",
        Assert { .. } => "Assert",
        Yield { .. } => "Yield",
        GeneratorDrop => "GeneratorDrop",
        FalseEdge { .. } => "FalseEdge",
        FalseUnwind { .. } => "FalseUnwind",
        InlineAsm { .. } => "InlineAsm",
    }
}

fn statement_span_viewable<'tcx>(
    tcx: TyCtxt<'tcx>,
    body_span: Span,
    bb: BasicBlock,
    i: usize,
    statement: &Statement<'tcx>,
) -> Option<SpanViewable> {
    let span = statement.source_info.span;
    if !body_span.contains(span) {
        return None;
    }
    let title = format!("bb{}[{}]", bb.index(), i);
    let tooltip = tooltip(tcx, &title, span, vec![statement.clone()], &None);
    Some(SpanViewable { span, title, tooltip })
}

fn terminator_span_viewable<'tcx>(
    tcx: TyCtxt<'tcx>,
    body_span: Span,
    bb: BasicBlock,
    data: &BasicBlockData<'tcx>,
) -> Option<SpanViewable> {
    let term = data.terminator();
    let span = term.source_info.span;
    if !body_span.contains(span) {
        return None;
    }
    let title = format!("bb{}`{}`", bb.index(), terminator_kind_name(term));
    let tooltip = tooltip(tcx, &title, span, vec![], &data.terminator);
    Some(SpanViewable { span, title, tooltip })
}

fn block_span_viewable<'tcx>(
    tcx: TyCtxt<'tcx>,
    body_span: Span,
    bb: BasicBlock,
    data: &BasicBlockData<'tcx>,
) -> Option<SpanViewable> {
    let span = compute_block_span(data, body_span);
    if !body_span.contains(span) {
        return None;
    }
    let title = format!("bb{}", bb.index());
    let tooltip = tooltip(tcx, &title, span, data.statements.clone(), &data.terminator);
    Some(SpanViewable { span, title, tooltip })
}

fn compute_block_span<'tcx>(data: &BasicBlockData<'tcx>, body_span: Span) -> Span {
    let mut span = data.terminator().source_info.span;
    for statement_span in data.statements.iter().map(|statement| statement.source_info.span) {
        // Only combine Spans from the function's body_span.
        if body_span.contains(statement_span) {
            span = span.to(statement_span);
        }
    }
    span
}

/// Recursively process each ordered span. Spans that overlap will have progressively varying
/// styles, such as increased padding for each overlap. Non-overlapping adjacent spans will
/// have alternating style choices, to help distinguish between them if, visually adjacent.
/// The `layer` is incremented for each overlap, and the `alt` bool alternates between true
/// and false, for each adjacent non-overlapping span. Source code between the spans (code
/// that is not in any coverage region) has neutral styling.
fn write_span_viewables<'tcx, 'b, W>(
    tcx: TyCtxt<'tcx>,
    next_pos: BytePos,
    ordered_span_viewables: &mut Peekable<impl Iterator<Item = &'b SpanViewable>>,
    alt: bool,
    layer: usize,
    w: &mut W,
) -> io::Result<BytePos>
where
    W: Write,
{
    let span_viewable =
        ordered_span_viewables.next().expect("ordered_span_viewables should have some");
    if next_pos < span_viewable.span.lo() {
        write_coverage_gap(tcx, next_pos, span_viewable.span.lo(), w)?;
    }
    let mut remaining_span = span_viewable.span;
    let mut subalt = false;
    loop {
        let next_span_viewable = match ordered_span_viewables.peek() {
            None => break,
            Some(span_viewable) => *span_viewable,
        };
        if !next_span_viewable.span.overlaps(remaining_span) {
            break;
        }
        write_span(
            tcx,
            remaining_span.until(next_span_viewable.span),
            Some(span_viewable),
            alt,
            layer,
            w,
        )?;
        let next_pos = write_span_viewables(
            tcx,
            next_span_viewable.span.lo(),
            ordered_span_viewables,
            subalt,
            layer + 1,
            w,
        )?;
        subalt = !subalt;
        if next_pos < remaining_span.hi() {
            remaining_span = remaining_span.with_lo(next_pos);
        } else {
            return Ok(next_pos);
        }
    }
    write_span(tcx, remaining_span, Some(span_viewable), alt, layer, w)
}

fn write_coverage_gap<'tcx, W>(
    tcx: TyCtxt<'tcx>,
    lo: BytePos,
    hi: BytePos,
    w: &mut W,
) -> io::Result<BytePos>
where
    W: Write,
{
    write_span(tcx, Span::with_root_ctxt(lo, hi), None, false, 0, w)
}

fn write_span<'tcx, W>(
    tcx: TyCtxt<'tcx>,
    span: Span,
    span_viewable: Option<&SpanViewable>,
    alt: bool,
    layer: usize,
    w: &mut W,
) -> io::Result<BytePos>
where
    W: Write,
{
    let source_map = tcx.sess.source_map();
    let snippet = source_map
        .span_to_snippet(span)
        .unwrap_or_else(|err| bug!("span_to_snippet error for span {:?}: {:?}", span, err));
    let labeled_snippet = if let Some(SpanViewable { title, .. }) = span_viewable {
        if span.is_empty() {
            format!(r#"<span class="annotation">@{}</span>"#, title)
        } else {
            format!(r#"<span class="annotation">@{}:</span> {}"#, title, escape_html(&snippet))
        }
    } else {
        snippet
    };
    let maybe_alt = if layer > 0 {
        if alt { " odd" } else { " even" }
    } else {
        ""
    };
    let maybe_tooltip = if let Some(SpanViewable { tooltip, .. }) = span_viewable {
        format!(" title=\"{}\"", escape_attr(tooltip))
    } else {
        "".to_owned()
    };
    if layer == 1 {
        write!(w, "<span>")?;
    }
    for (i, line) in labeled_snippet.lines().enumerate() {
        if i > 0 {
            write!(w, "{}", NEW_LINE_SPAN)?;
        }
        write!(
            w,
            r#"<span class="code{}" style="--layer: {}"{}>{}</span>"#,
            maybe_alt, layer, maybe_tooltip, line
        )?;
    }
    if layer == 1 {
        write!(w, "</span>")?;
    }
    Ok(span.hi())
}

fn tooltip<'tcx>(
    tcx: TyCtxt<'tcx>,
    title: &str,
    span: Span,
    statements: Vec<Statement<'tcx>>,
    terminator: &Option<Terminator<'tcx>>,
) -> String {
    let source_map = tcx.sess.source_map();
    let mut text = Vec::new();
    text.push(format!("{}: {}:", title, &source_map.span_to_string(span)));
    for statement in statements {
        let source_range = source_range_no_file(tcx, &statement.source_info.span);
        text.push(format!(
            "\n{}{}: {}: {}",
            TOOLTIP_INDENT,
            source_range,
            statement_kind_name(&statement),
            format!("{:?}", statement)
        ));
    }
    if let Some(term) = terminator {
        let source_range = source_range_no_file(tcx, &term.source_info.span);
        text.push(format!(
            "\n{}{}: {}: {:?}",
            TOOLTIP_INDENT,
            source_range,
            terminator_kind_name(term),
            term.kind
        ));
    }
    text.join("")
}

fn fn_span<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> Span {
    let hir_id =
        tcx.hir().local_def_id_to_hir_id(def_id.as_local().expect("expected DefId is local"));
    tcx.hir().span(hir_id)
}

fn hir_body<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> &'tcx rustc_hir::Body<'tcx> {
    let hir_node = tcx.hir().get_if_local(def_id).expect("expected DefId is local");
    let fn_body_id = hir::map::associated_body(hir_node).expect("HIR node is a function with body");
    tcx.hir().body(fn_body_id)
}

fn escape_html(s: &str) -> String {
    s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
}

fn escape_attr(s: &str) -> String {
    s.replace("&", "&amp;")
        .replace("\"", "&quot;")
        .replace("'", "&#39;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
}
