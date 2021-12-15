use rustc_hir::def_id::DefId;
use rustc_middle::hir;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::MirSpanview;
use rustc_span::{BytePos, Pos, Span, SyntaxContext};

use std::cmp;
use std::io::{self, Write};

pub const TOOLTIP_INDENT: &str = "    ";

const CARET: char = '\u{2038}'; // Unicode `CARET`
const ANNOTATION_LEFT_BRACKET: char = '\u{298a}'; // Unicode `Z NOTATION RIGHT BINDING BRACKET
const ANNOTATION_RIGHT_BRACKET: char = '\u{2989}'; // Unicode `Z NOTATION LEFT BINDING BRACKET`
const NEW_LINE_SPAN: &str = "</span>\n<span class=\"line\">";
const HEADER: &str = r#"<!DOCTYPE html>
<html>
<head>"#;
const START_BODY: &str = r#"</head>
<body>"#;
const FOOTER: &str = r#"</body>
</html>"#;

const STYLE_SECTION: &str = r#"<style>
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
</style>"#;

/// Metadata to highlight the span of a MIR BasicBlock, Statement, or Terminator.
#[derive(Clone, Debug)]
pub struct SpanViewable {
    pub bb: BasicBlock,
    pub span: Span,
    pub id: String,
    pub tooltip: String,
}

/// Write a spanview HTML+CSS file to analyze MIR element spans.
pub fn write_mir_fn_spanview<'tcx, W>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    spanview: MirSpanview,
    title: &str,
    w: &mut W,
) -> io::Result<()>
where
    W: Write,
{
    let def_id = body.source.def_id();
    let hir_body = hir_body(tcx, def_id);
    if hir_body.is_none() {
        return Ok(());
    }
    let body_span = hir_body.unwrap().value.span;
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
    write_document(tcx, fn_span(tcx, def_id), span_viewables, title, w)?;
    Ok(())
}

/// Generate a spanview HTML+CSS document for the given local function `def_id`, and a pre-generated
/// list `SpanViewable`s.
pub fn write_document<'tcx, W>(
    tcx: TyCtxt<'tcx>,
    spanview_span: Span,
    mut span_viewables: Vec<SpanViewable>,
    title: &str,
    w: &mut W,
) -> io::Result<()>
where
    W: Write,
{
    let mut from_pos = spanview_span.lo();
    let end_pos = spanview_span.hi();
    let source_map = tcx.sess.source_map();
    let start = source_map.lookup_char_pos(from_pos);
    let indent_to_initial_start_col = " ".repeat(start.col.to_usize());
    debug!(
        "spanview_span={:?}; source is:\n{}{}",
        spanview_span,
        indent_to_initial_start_col,
        source_map.span_to_snippet(spanview_span).expect("function should have printable source")
    );
    writeln!(w, "{}", HEADER)?;
    writeln!(w, "<title>{}</title>", title)?;
    writeln!(w, "{}", STYLE_SECTION)?;
    writeln!(w, "{}", START_BODY)?;
    write!(
        w,
        r#"<div class="code" style="counter-reset: line {}"><span class="line">{}"#,
        start.line - 1,
        indent_to_initial_start_col,
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
    let mut ordered_viewables = &span_viewables[..];
    const LOWEST_VIEWABLE_LAYER: usize = 1;
    let mut alt = false;
    while ordered_viewables.len() > 0 {
        debug!(
            "calling write_next_viewable with from_pos={}, end_pos={}, and viewables len={}",
            from_pos.to_usize(),
            end_pos.to_usize(),
            ordered_viewables.len()
        );
        let curr_id = &ordered_viewables[0].id;
        let (next_from_pos, next_ordered_viewables) = write_next_viewable_with_overlaps(
            tcx,
            from_pos,
            end_pos,
            ordered_viewables,
            alt,
            LOWEST_VIEWABLE_LAYER,
            w,
        )?;
        debug!(
            "DONE calling write_next_viewable, with new from_pos={}, \
             and remaining viewables len={}",
            next_from_pos.to_usize(),
            next_ordered_viewables.len()
        );
        assert!(
            from_pos != next_from_pos || ordered_viewables.len() != next_ordered_viewables.len(),
            "write_next_viewable_with_overlaps() must make a state change"
        );
        from_pos = next_from_pos;
        if next_ordered_viewables.len() != ordered_viewables.len() {
            ordered_viewables = next_ordered_viewables;
            if let Some(next_ordered_viewable) = ordered_viewables.first() {
                if &next_ordered_viewable.id != curr_id {
                    alt = !alt;
                }
            }
        }
    }
    if from_pos < end_pos {
        write_coverage_gap(tcx, from_pos, end_pos, w)?;
    }
    writeln!(w, r#"</span></div>"#)?;
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
        CopyNonOverlapping(..) => "CopyNonOverlapping",
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
    let id = format!("{}[{}]", bb.index(), i);
    let tooltip = tooltip(tcx, &id, span, vec![statement.clone()], &None);
    Some(SpanViewable { bb, span, id, tooltip })
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
    let id = format!("{}:{}", bb.index(), terminator_kind_name(term));
    let tooltip = tooltip(tcx, &id, span, vec![], &data.terminator);
    Some(SpanViewable { bb, span, id, tooltip })
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
    let id = format!("{}", bb.index());
    let tooltip = tooltip(tcx, &id, span, data.statements.clone(), &data.terminator);
    Some(SpanViewable { bb, span, id, tooltip })
}

fn compute_block_span<'tcx>(data: &BasicBlockData<'tcx>, body_span: Span) -> Span {
    let mut span = data.terminator().source_info.span;
    for statement_span in data.statements.iter().map(|statement| statement.source_info.span) {
        // Only combine Spans from the root context, and within the function's body_span.
        if statement_span.ctxt() == SyntaxContext::root() && body_span.contains(statement_span) {
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
fn write_next_viewable_with_overlaps<'tcx, 'b, W>(
    tcx: TyCtxt<'tcx>,
    mut from_pos: BytePos,
    mut to_pos: BytePos,
    ordered_viewables: &'b [SpanViewable],
    alt: bool,
    layer: usize,
    w: &mut W,
) -> io::Result<(BytePos, &'b [SpanViewable])>
where
    W: Write,
{
    let debug_indent = "  ".repeat(layer);
    let (viewable, mut remaining_viewables) =
        ordered_viewables.split_first().expect("ordered_viewables should have some");

    if from_pos < viewable.span.lo() {
        debug!(
            "{}advance from_pos to next SpanViewable (from from_pos={} to viewable.span.lo()={} \
             of {:?}), with to_pos={}",
            debug_indent,
            from_pos.to_usize(),
            viewable.span.lo().to_usize(),
            viewable.span,
            to_pos.to_usize()
        );
        let hi = cmp::min(viewable.span.lo(), to_pos);
        write_coverage_gap(tcx, from_pos, hi, w)?;
        from_pos = hi;
        if from_pos < viewable.span.lo() {
            debug!(
                "{}EARLY RETURN: stopped before getting to next SpanViewable, at {}",
                debug_indent,
                from_pos.to_usize()
            );
            return Ok((from_pos, ordered_viewables));
        }
    }

    if from_pos < viewable.span.hi() {
        // Set to_pos to the end of this `viewable` to ensure the recursive calls stop writing
        // with room to print the tail.
        to_pos = cmp::min(viewable.span.hi(), to_pos);
        debug!(
            "{}update to_pos (if not closer) to viewable.span.hi()={}; to_pos is now {}",
            debug_indent,
            viewable.span.hi().to_usize(),
            to_pos.to_usize()
        );
    }

    let mut subalt = false;
    while remaining_viewables.len() > 0 && remaining_viewables[0].span.overlaps(viewable.span) {
        let overlapping_viewable = &remaining_viewables[0];
        debug!("{}overlapping_viewable.span={:?}", debug_indent, overlapping_viewable.span);

        let span =
            trim_span(viewable.span, from_pos, cmp::min(overlapping_viewable.span.lo(), to_pos));
        let mut some_html_snippet = if from_pos <= viewable.span.hi() || viewable.span.is_empty() {
            // `viewable` is not yet fully rendered, so start writing the span, up to either the
            // `to_pos` or the next `overlapping_viewable`, whichever comes first.
            debug!(
                "{}make html_snippet (may not write it if early exit) for partial span {:?} \
                 of viewable.span {:?}",
                debug_indent, span, viewable.span
            );
            from_pos = span.hi();
            make_html_snippet(tcx, span, Some(&viewable))
        } else {
            None
        };

        // Defer writing the HTML snippet (until after early return checks) ONLY for empty spans.
        // An empty Span with Some(html_snippet) is probably a tail marker. If there is an early
        // exit, there should be another opportunity to write the tail marker.
        if !span.is_empty() {
            if let Some(ref html_snippet) = some_html_snippet {
                debug!(
                    "{}write html_snippet for that partial span of viewable.span {:?}",
                    debug_indent, viewable.span
                );
                write_span(html_snippet, &viewable.tooltip, alt, layer, w)?;
            }
            some_html_snippet = None;
        }

        if from_pos < overlapping_viewable.span.lo() {
            debug!(
                "{}EARLY RETURN: from_pos={} has not yet reached the \
                 overlapping_viewable.span {:?}",
                debug_indent,
                from_pos.to_usize(),
                overlapping_viewable.span
            );
            // must have reached `to_pos` before reaching the start of the
            // `overlapping_viewable.span`
            return Ok((from_pos, ordered_viewables));
        }

        if from_pos == to_pos
            && !(from_pos == overlapping_viewable.span.lo() && overlapping_viewable.span.is_empty())
        {
            debug!(
                "{}EARLY RETURN: from_pos=to_pos={} and overlapping_viewable.span {:?} is not \
                 empty, or not from_pos",
                debug_indent,
                to_pos.to_usize(),
                overlapping_viewable.span
            );
            // `to_pos` must have occurred before the overlapping viewable. Return
            // `ordered_viewables` so we can continue rendering the `viewable`, from after the
            // `to_pos`.
            return Ok((from_pos, ordered_viewables));
        }

        if let Some(ref html_snippet) = some_html_snippet {
            debug!(
                "{}write html_snippet for that partial span of viewable.span {:?}",
                debug_indent, viewable.span
            );
            write_span(html_snippet, &viewable.tooltip, alt, layer, w)?;
        }

        debug!(
            "{}recursively calling write_next_viewable with from_pos={}, to_pos={}, \
             and viewables len={}",
            debug_indent,
            from_pos.to_usize(),
            to_pos.to_usize(),
            remaining_viewables.len()
        );
        // Write the overlaps (and the overlaps' overlaps, if any) up to `to_pos`.
        let curr_id = &remaining_viewables[0].id;
        let (next_from_pos, next_remaining_viewables) = write_next_viewable_with_overlaps(
            tcx,
            from_pos,
            to_pos,
            &remaining_viewables,
            subalt,
            layer + 1,
            w,
        )?;
        debug!(
            "{}DONE recursively calling write_next_viewable, with new from_pos={}, and remaining \
             viewables len={}",
            debug_indent,
            next_from_pos.to_usize(),
            next_remaining_viewables.len()
        );
        assert!(
            from_pos != next_from_pos
                || remaining_viewables.len() != next_remaining_viewables.len(),
            "write_next_viewable_with_overlaps() must make a state change"
        );
        from_pos = next_from_pos;
        if next_remaining_viewables.len() != remaining_viewables.len() {
            remaining_viewables = next_remaining_viewables;
            if let Some(next_ordered_viewable) = remaining_viewables.first() {
                if &next_ordered_viewable.id != curr_id {
                    subalt = !subalt;
                }
            }
        }
    }
    if from_pos <= viewable.span.hi() {
        let span = trim_span(viewable.span, from_pos, to_pos);
        debug!(
            "{}After overlaps, writing (end span?) {:?} of viewable.span {:?}",
            debug_indent, span, viewable.span
        );
        if let Some(ref html_snippet) = make_html_snippet(tcx, span, Some(&viewable)) {
            from_pos = span.hi();
            write_span(html_snippet, &viewable.tooltip, alt, layer, w)?;
        }
    }
    debug!("{}RETURN: No more overlap", debug_indent);
    Ok((
        from_pos,
        if from_pos < viewable.span.hi() { ordered_viewables } else { remaining_viewables },
    ))
}

#[inline(always)]
fn write_coverage_gap<'tcx, W>(
    tcx: TyCtxt<'tcx>,
    lo: BytePos,
    hi: BytePos,
    w: &mut W,
) -> io::Result<()>
where
    W: Write,
{
    let span = Span::with_root_ctxt(lo, hi);
    if let Some(ref html_snippet) = make_html_snippet(tcx, span, None) {
        write_span(html_snippet, "", false, 0, w)
    } else {
        Ok(())
    }
}

fn write_span<W>(
    html_snippet: &str,
    tooltip: &str,
    alt: bool,
    layer: usize,
    w: &mut W,
) -> io::Result<()>
where
    W: Write,
{
    let maybe_alt_class = if layer > 0 {
        if alt { " odd" } else { " even" }
    } else {
        ""
    };
    let maybe_title_attr = if !tooltip.is_empty() {
        format!(" title=\"{}\"", escape_attr(tooltip))
    } else {
        "".to_owned()
    };
    if layer == 1 {
        write!(w, "<span>")?;
    }
    for (i, line) in html_snippet.lines().enumerate() {
        if i > 0 {
            write!(w, "{}", NEW_LINE_SPAN)?;
        }
        write!(
            w,
            r#"<span class="code{}" style="--layer: {}"{}>{}</span>"#,
            maybe_alt_class, layer, maybe_title_attr, line
        )?;
    }
    // Check for and translate trailing newlines, because `str::lines()` ignores them
    if html_snippet.ends_with('\n') {
        write!(w, "{}", NEW_LINE_SPAN)?;
    }
    if layer == 1 {
        write!(w, "</span>")?;
    }
    Ok(())
}

fn make_html_snippet<'tcx>(
    tcx: TyCtxt<'tcx>,
    span: Span,
    some_viewable: Option<&SpanViewable>,
) -> Option<String> {
    let source_map = tcx.sess.source_map();
    let snippet = source_map
        .span_to_snippet(span)
        .unwrap_or_else(|err| bug!("span_to_snippet error for span {:?}: {:?}", span, err));
    let html_snippet = if let Some(viewable) = some_viewable {
        let is_head = span.lo() == viewable.span.lo();
        let is_tail = span.hi() == viewable.span.hi();
        let mut labeled_snippet = if is_head {
            format!(r#"<span class="annotation">{}{}</span>"#, viewable.id, ANNOTATION_LEFT_BRACKET)
        } else {
            "".to_owned()
        };
        if span.is_empty() {
            if is_head && is_tail {
                labeled_snippet.push(CARET);
            }
        } else {
            labeled_snippet.push_str(&escape_html(&snippet));
        };
        if is_tail {
            labeled_snippet.push_str(&format!(
                r#"<span class="annotation">{}{}</span>"#,
                ANNOTATION_RIGHT_BRACKET, viewable.id
            ));
        }
        labeled_snippet
    } else {
        escape_html(&snippet)
    };
    if html_snippet.is_empty() { None } else { Some(html_snippet) }
}

fn tooltip<'tcx>(
    tcx: TyCtxt<'tcx>,
    spanview_id: &str,
    span: Span,
    statements: Vec<Statement<'tcx>>,
    terminator: &Option<Terminator<'tcx>>,
) -> String {
    let source_map = tcx.sess.source_map();
    let mut text = Vec::new();
    text.push(format!("{}: {}:", spanview_id, &source_map.span_to_embeddable_string(span)));
    for statement in statements {
        let source_range = source_range_no_file(tcx, &statement.source_info.span);
        text.push(format!(
            "\n{}{}: {}: {:?}",
            TOOLTIP_INDENT,
            source_range,
            statement_kind_name(&statement),
            statement
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

fn trim_span(span: Span, from_pos: BytePos, to_pos: BytePos) -> Span {
    trim_span_hi(trim_span_lo(span, from_pos), to_pos)
}

fn trim_span_lo(span: Span, from_pos: BytePos) -> Span {
    if from_pos <= span.lo() { span } else { span.with_lo(cmp::min(span.hi(), from_pos)) }
}

fn trim_span_hi(span: Span, to_pos: BytePos) -> Span {
    if to_pos >= span.hi() { span } else { span.with_hi(cmp::max(span.lo(), to_pos)) }
}

fn fn_span<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> Span {
    let hir_id =
        tcx.hir().local_def_id_to_hir_id(def_id.as_local().expect("expected DefId is local"));
    let fn_decl_span = tcx.hir().span(hir_id);
    if let Some(body_span) = hir_body(tcx, def_id).map(|hir_body| hir_body.value.span) {
        if fn_decl_span.ctxt() == body_span.ctxt() { fn_decl_span.to(body_span) } else { body_span }
    } else {
        fn_decl_span
    }
}

fn hir_body<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> Option<&'tcx rustc_hir::Body<'tcx>> {
    let hir_node = tcx.hir().get_if_local(def_id).expect("expected DefId is local");
    hir::map::associated_body(hir_node).map(|fn_body_id| tcx.hir().body(fn_body_id))
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn escape_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('\"', "&quot;")
        .replace('\'', "&#39;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
