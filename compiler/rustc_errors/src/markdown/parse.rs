use crate::markdown::{MdStream, MdTree};
use std::{iter, mem, str};

/// Short aliases that we can use in match patterns. If an end pattern is not
/// included, this type may be variable
const ANC_E: &[u8] = b">";
const ANC_S: &[u8] = b"<";
const BRK: &[u8] = b"---";
const CBK: &[u8] = b"```";
const CIL: &[u8] = b"`";
const CMT_E: &[u8] = b"-->";
const CMT_S: &[u8] = b"<!--";
const EMP: &[u8] = b"_";
const HDG: &[u8] = b"#";
const LNK_CHARS: &str = "$-_.+!*'()/&?=:%";
const LNK_E: &[u8] = b"]";
const LNK_S: &[u8] = b"[";
const STG: &[u8] = b"**";
const STK: &[u8] = b"~~";
const UL1: &[u8] = b"* ";
const UL2: &[u8] = b"- ";

/// Pattern replacements
const REPLACEMENTS: &[(&str, &str)] = &[
    ("(c)", "©"),
    ("(C)", "©"),
    ("(r)", "®"),
    ("(R)", "®"),
    ("(tm)", "™"),
    ("(TM)", "™"),
    (":crab:", "🦀"),
    ("\n", " "),
];

/// `(extracted, remaining)`
type Parsed<'a> = (MdTree<'a>, &'a [u8]);
/// Output of a parse function
type ParseResult<'a> = Option<Parsed<'a>>;

/// Parsing context
#[derive(Clone, Copy, Debug, PartialEq)]
struct Context {
    /// If true, we are at a the topmost level (not recursing a nested tt)
    top_block: bool,
    /// Previous character
    prev: Prev,
}

/// Character class preceding this one
#[derive(Clone, Copy, Debug, PartialEq)]
enum Prev {
    Newline,
    /// Whitespace that is not a newline
    Whitespace,
    Escape,
    Any,
}

impl Default for Context {
    /// Most common setting for non top-level parsing: not top block, not at
    /// line start (yes leading whitespace, not escaped)
    fn default() -> Self {
        Self { top_block: false, prev: Prev::Whitespace }
    }
}

/// Flags to simple parser function
#[derive(Clone, Copy, Debug, PartialEq)]
enum ParseOpt {
    /// Ignore escapes before closing pattern, trim content
    TrimNoEsc,
    None,
}

/// Parse a buffer
pub fn entrypoint(txt: &str) -> MdStream<'_> {
    let ctx = Context { top_block: true, prev: Prev::Newline };
    normalize(parse_recursive(txt.trim().as_bytes(), ctx), &mut Vec::new())
}

/// Parse a buffer with specified context
fn parse_recursive<'a>(buf: &'a [u8], ctx: Context) -> MdStream<'_> {
    use ParseOpt as Po;
    use Prev::{Escape, Newline, Whitespace};

    let mut stream: Vec<MdTree<'a>> = Vec::new();
    let Context { top_block: top_blk, mut prev } = ctx;

    // wip_buf is our entire unprocessed (unpushed) buffer, loop_buf is our to
    // check buffer that shrinks with each loop
    let mut wip_buf = buf;
    let mut loop_buf = wip_buf;

    while !loop_buf.is_empty() {
        let next_prev = match loop_buf[0] {
            b'\n' => Newline,
            b'\\' => Escape,
            x if x.is_ascii_whitespace() => Whitespace,
            _ => Prev::Any,
        };

        let res: ParseResult<'_> = match (top_blk, prev) {
            (_, Newline | Whitespace) if loop_buf.starts_with(CMT_S) => {
                parse_simple_pat(loop_buf, CMT_S, CMT_E, Po::TrimNoEsc, MdTree::Comment)
            }
            (true, Newline) if loop_buf.starts_with(CBK) => Some(parse_codeblock(loop_buf)),
            (_, Newline | Whitespace) if loop_buf.starts_with(CIL) => parse_codeinline(loop_buf),
            (true, Newline | Whitespace) if loop_buf.starts_with(HDG) => parse_heading(loop_buf),
            (true, Newline) if loop_buf.starts_with(BRK) => {
                Some((MdTree::HorizontalRule, parse_to_newline(loop_buf).1))
            }
            (_, Newline | Whitespace) if loop_buf.starts_with(EMP) => {
                parse_simple_pat(loop_buf, EMP, EMP, Po::None, MdTree::Emphasis)
            }
            (_, Newline | Whitespace) if loop_buf.starts_with(STG) => {
                parse_simple_pat(loop_buf, STG, STG, Po::None, MdTree::Strong)
            }
            (_, Newline | Whitespace) if loop_buf.starts_with(STK) => {
                parse_simple_pat(loop_buf, STK, STK, Po::None, MdTree::Strikethrough)
            }
            (_, Newline | Whitespace) if loop_buf.starts_with(ANC_S) => {
                let tt_fn = |link| MdTree::Link { disp: link, link };
                let ret = parse_simple_pat(loop_buf, ANC_S, ANC_E, Po::None, tt_fn);
                match ret {
                    Some((MdTree::Link { disp, .. }, _))
                        if disp.chars().all(|ch| LNK_CHARS.contains(ch)) =>
                    {
                        ret
                    }
                    _ => None,
                }
            }
            (_, Newline) if (loop_buf.starts_with(UL1) || loop_buf.starts_with(UL2)) => {
                Some(parse_unordered_li(loop_buf))
            }
            (_, Newline) if ord_list_start(loop_buf).is_some() => Some(parse_ordered_li(loop_buf)),
            (_, Newline | Whitespace) if loop_buf.starts_with(LNK_S) => {
                parse_any_link(loop_buf, top_blk && prev == Prev::Newline)
            }
            (_, Escape | _) => None,
        };

        if let Some((tree, rest)) = res {
            // We found something: push our WIP and then push the found tree
            let prev_buf = &wip_buf[..(wip_buf.len() - loop_buf.len())];
            if !prev_buf.is_empty() {
                let prev_str = str::from_utf8(prev_buf).unwrap();
                stream.push(MdTree::PlainText(prev_str));
            }
            stream.push(tree);

            wip_buf = rest;
            loop_buf = rest;
        } else {
            // Just move on to the next character
            loop_buf = &loop_buf[1..];
            // If we are at the end and haven't found anything, just push plain text
            if loop_buf.is_empty() && !wip_buf.is_empty() {
                let final_str = str::from_utf8(wip_buf).unwrap();
                stream.push(MdTree::PlainText(final_str));
            }
        };

        prev = next_prev;
    }

    MdStream(stream)
}

/// The simplest kind of patterns: data within start and end patterns
fn parse_simple_pat<'a, F>(
    buf: &'a [u8],
    start_pat: &[u8],
    end_pat: &[u8],
    opts: ParseOpt,
    create_tt: F,
) -> ParseResult<'a>
where
    F: FnOnce(&'a str) -> MdTree<'a>,
{
    let ignore_esc = matches!(opts, ParseOpt::TrimNoEsc);
    let trim = matches!(opts, ParseOpt::TrimNoEsc);
    let (txt, rest) = parse_with_end_pat(&buf[start_pat.len()..], end_pat, ignore_esc)?;
    let mut txt = str::from_utf8(txt).unwrap();
    if trim {
        txt = txt.trim();
    }
    Some((create_tt(txt), rest))
}

/// Parse backtick-wrapped inline code. Accounts for >1 backtick sets
fn parse_codeinline(buf: &[u8]) -> ParseResult<'_> {
    let seps = buf.iter().take_while(|ch| **ch == b'`').count();
    let (txt, rest) = parse_with_end_pat(&buf[seps..], &buf[..seps], true)?;
    Some((MdTree::CodeInline(str::from_utf8(txt).unwrap()), rest))
}

/// Parse a codeblock. Accounts for >3 backticks and language specification
fn parse_codeblock(buf: &[u8]) -> Parsed<'_> {
    // account for ````code```` style
    let seps = buf.iter().take_while(|ch| **ch == b'`').count();
    let end_sep = &buf[..seps];
    let mut working = &buf[seps..];

    // Handle "````rust" style language specifications
    let next_ws_idx = working.iter().take_while(|ch| !ch.is_ascii_whitespace()).count();

    let lang = if next_ws_idx > 0 {
        // Munch the lang
        let tmp = str::from_utf8(&working[..next_ws_idx]).unwrap();
        working = &working[next_ws_idx..];
        Some(tmp)
    } else {
        None
    };

    let mut end_pat = vec![b'\n'];
    end_pat.extend(end_sep);

    // Find first end pattern with nothing else on its line
    let mut found = None;
    for idx in (0..working.len()).filter(|idx| working[*idx..].starts_with(&end_pat)) {
        let (eol_txt, rest) = parse_to_newline(&working[(idx + end_pat.len())..]);
        if !eol_txt.iter().any(u8::is_ascii_whitespace) {
            found = Some((&working[..idx], rest));
            break;
        }
    }

    let (txt, rest) = found.unwrap_or((working, &[]));
    let txt = str::from_utf8(txt).unwrap().trim_matches('\n');

    (MdTree::CodeBlock { txt, lang }, rest)
}

fn parse_heading(buf: &[u8]) -> ParseResult<'_> {
    let level = buf.iter().take_while(|ch| **ch == b'#').count();
    let buf = &buf[level..];

    if level > 6 || (buf.len() > 1 && !buf[0].is_ascii_whitespace()) {
        // Enforce max 6 levels and whitespace following the `##` pattern
        return None;
    }

    let (txt, rest) = parse_to_newline(&buf[1..]);
    let ctx = Context { top_block: false, prev: Prev::Whitespace };
    let stream = parse_recursive(txt, ctx);

    Some((MdTree::Heading(level.try_into().unwrap(), stream), rest))
}

/// Bulleted list
fn parse_unordered_li(buf: &[u8]) -> Parsed<'_> {
    debug_assert!(buf.starts_with(b"* ") || buf.starts_with(b"- "));
    let (txt, rest) = get_indented_section(&buf[2..]);
    let ctx = Context { top_block: false, prev: Prev::Whitespace };
    let stream = parse_recursive(trim_ascii_start(txt), ctx);
    (MdTree::UnorderedListItem(stream), rest)
}

/// Numbered list
fn parse_ordered_li(buf: &[u8]) -> Parsed<'_> {
    let (num, pos) = ord_list_start(buf).unwrap(); // success tested in caller
    let (txt, rest) = get_indented_section(&buf[pos..]);
    let ctx = Context { top_block: false, prev: Prev::Whitespace };
    let stream = parse_recursive(trim_ascii_start(txt), ctx);
    (MdTree::OrderedListItem(num, stream), rest)
}

/// Find first line that isn't empty or doesn't start with whitespace, that will
/// be our contents
fn get_indented_section(buf: &[u8]) -> (&[u8], &[u8]) {
    let mut end = buf.len();
    for (idx, window) in buf.windows(2).enumerate() {
        let &[ch, next_ch] = window else {unreachable!("always 2 elements")};
        if idx >= buf.len().saturating_sub(2) && next_ch == b'\n' {
            // End of stream
            end = buf.len().saturating_sub(1);
            break;
        } else if ch == b'\n' && (!next_ch.is_ascii_whitespace() || next_ch == b'\n') {
            end = idx;
            break;
        }
    }

    (&buf[..end], &buf[end..])
}

/// Verify a valid ordered list start (e.g. `1.`) and parse it. Returns the
/// parsed number and offset of character after the dot.
fn ord_list_start(buf: &[u8]) -> Option<(u16, usize)> {
    let pos = buf.iter().take(10).position(|ch| *ch == b'.')?;
    let n = str::from_utf8(&buf[..pos]).ok()?;
    if !buf.get(pos + 1)?.is_ascii_whitespace() {
        return None;
    }
    n.parse::<u16>().ok().map(|v| (v, pos + 2))
}

/// Parse links. `can_be_def` indicates that a link definition is possible (top
/// level, located at the start of a line)
fn parse_any_link(buf: &[u8], can_be_def: bool) -> ParseResult<'_> {
    let (bracketed, rest) = parse_with_end_pat(&buf[1..], LNK_E, true)?;
    if rest.is_empty() {
        return None;
    }

    let disp = str::from_utf8(bracketed).unwrap();
    match (can_be_def, rest[0]) {
        (true, b':') => {
            let (link, tmp) = parse_to_newline(&rest[1..]);
            let link = str::from_utf8(link).unwrap().trim();
            Some((MdTree::LinkDef { id: disp, link }, tmp))
        }
        (_, b'(') => parse_simple_pat(rest, b"(", b")", ParseOpt::TrimNoEsc, |link| MdTree::Link {
            disp,
            link,
        }),
        (_, b'[') => parse_simple_pat(rest, b"[", b"]", ParseOpt::TrimNoEsc, |id| {
            MdTree::RefLink { disp, id: Some(id) }
        }),
        _ => Some((MdTree::RefLink { disp, id: None }, rest)),
    }
}

/// Find and consume an end pattern, return `(match, residual)`
fn parse_with_end_pat<'a>(
    buf: &'a [u8],
    end_sep: &[u8],
    ignore_esc: bool,
) -> Option<(&'a [u8], &'a [u8])> {
    // Find positions that start with the end seperator
    for idx in (0..buf.len()).filter(|idx| buf[*idx..].starts_with(end_sep)) {
        if !ignore_esc && idx > 0 && buf[idx - 1] == b'\\' {
            continue;
        }
        return Some((&buf[..idx], &buf[idx + end_sep.len()..]));
    }
    None
}

/// Resturn `(match, residual)` to end of line. The EOL is returned with the
/// residual.
fn parse_to_newline(buf: &[u8]) -> (&[u8], &[u8]) {
    buf.iter().position(|ch| *ch == b'\n').map_or((buf, &[]), |pos| buf.split_at(pos))
}

/// Take a parsed stream and fix the little things
fn normalize<'a>(MdStream(stream): MdStream<'a>, linkdefs: &mut Vec<MdTree<'a>>) -> MdStream<'a> {
    let mut new_stream = Vec::with_capacity(stream.len());
    let new_defs = stream.iter().filter(|tt| matches!(tt, MdTree::LinkDef { .. }));
    linkdefs.extend(new_defs.cloned());

    // Run plaintest expansions on types that need it, call this function on nested types
    for item in stream {
        match item {
            MdTree::PlainText(txt) => expand_plaintext(txt, &mut new_stream, MdTree::PlainText),
            MdTree::Strong(txt) => expand_plaintext(txt, &mut new_stream, MdTree::Strong),
            MdTree::Emphasis(txt) => expand_plaintext(txt, &mut new_stream, MdTree::Emphasis),
            MdTree::Strikethrough(txt) => {
                expand_plaintext(txt, &mut new_stream, MdTree::Strikethrough);
            }
            MdTree::RefLink { disp, id } => new_stream.push(match_reflink(linkdefs, disp, id)),
            MdTree::OrderedListItem(n, st) => {
                new_stream.push(MdTree::OrderedListItem(n, normalize(st, linkdefs)));
            }
            MdTree::UnorderedListItem(st) => {
                new_stream.push(MdTree::UnorderedListItem(normalize(st, linkdefs)));
            }
            MdTree::Heading(n, st) => new_stream.push(MdTree::Heading(n, normalize(st, linkdefs))),
            _ => new_stream.push(item),
        }
    }

    // Remove non printing types, duplicate paragraph breaks, and breaks at start/end
    new_stream.retain(|x| !matches!(x, MdTree::Comment(_) | MdTree::LinkDef { .. }));
    new_stream.dedup_by(|r, l| matches!((r, l), (MdTree::ParagraphBreak, MdTree::ParagraphBreak)));

    if new_stream.first().is_some_and(is_break_ty) {
        new_stream.remove(0);
    }
    if new_stream.last().is_some_and(is_break_ty) {
        new_stream.pop();
    }

    // Remove paragraph breaks that shouldn't be there. w[1] is what will be
    // removed in these cases. Note that these are the items to keep, not delete
    // (for `retain`)
    let to_keep: Vec<bool> = new_stream
        .windows(3)
        .map(|w| {
            !((matches!(&w[1], MdTree::ParagraphBreak)
                && matches!(should_break(&w[0], &w[2]), BreakRule::Always(1) | BreakRule::Never))
                || (matches!(&w[1], MdTree::PlainText(txt) if txt.trim().is_empty())
                    && matches!(
                        should_break(&w[0], &w[2]),
                        BreakRule::Always(_) | BreakRule::Never
                    )))
        })
        .collect();
    let mut iter = iter::once(true).chain(to_keep).chain(iter::once(true));
    new_stream.retain(|_| iter.next().unwrap());

    // Insert line or paragraph breaks where there should be some
    let mut insertions = 0;
    let to_insert: Vec<(usize, MdTree<'_>)> = new_stream
        .windows(2)
        .enumerate()
        .filter_map(|(idx, w)| match should_break(&w[0], &w[1]) {
            BreakRule::Always(1) => Some((idx, MdTree::LineBreak)),
            BreakRule::Always(2) => Some((idx, MdTree::ParagraphBreak)),
            _ => None,
        })
        .map(|(idx, tt)| {
            insertions += 1;
            (idx + insertions, tt)
        })
        .collect();
    to_insert.into_iter().for_each(|(idx, tt)| new_stream.insert(idx, tt));

    MdStream(new_stream)
}

/// Whether two types should or shouldn't have a paragraph break between them
#[derive(Clone, Copy, Debug, PartialEq)]
enum BreakRule {
    Always(u8),
    Never,
    Optional,
}

/// Blocks that automatically handle their own text wrapping
fn should_break(left: &MdTree<'_>, right: &MdTree<'_>) -> BreakRule {
    use MdTree::*;

    match (left, right) {
        // Separate these types with a single line
        (HorizontalRule, _)
        | (_, HorizontalRule)
        | (OrderedListItem(_, _), OrderedListItem(_, _))
        | (UnorderedListItem(_), UnorderedListItem(_)) => BreakRule::Always(1),
        // Condensed types shouldn't have an extra break on either side
        (Comment(_) | ParagraphBreak | Heading(_, _), _) | (_, Comment(_) | ParagraphBreak) => {
            BreakRule::Never
        }
        // Block types should always be separated by full breaks
        (CodeBlock { .. } | OrderedListItem(_, _) | UnorderedListItem(_), _)
        | (_, CodeBlock { .. } | Heading(_, _) | OrderedListItem(_, _) | UnorderedListItem(_)) => {
            BreakRule::Always(2)
        }
        // Text types may or may not be separated by a break
        (
            CodeInline(_)
            | Strong(_)
            | Emphasis(_)
            | Strikethrough(_)
            | PlainText(_)
            | Link { .. }
            | RefLink { .. }
            | LinkDef { .. },
            CodeInline(_)
            | Strong(_)
            | Emphasis(_)
            | Strikethrough(_)
            | PlainText(_)
            | Link { .. }
            | RefLink { .. }
            | LinkDef { .. },
        ) => BreakRule::Optional,
        (LineBreak, _) | (_, LineBreak) => {
            unreachable!("should have been removed during deduplication")
        }
    }
}

/// Types that indicate some form of break
fn is_break_ty(val: &MdTree<'_>) -> bool {
    matches!(val, MdTree::ParagraphBreak | MdTree::LineBreak)
        // >1 break between paragraphs acts as a break
        || matches!(val, MdTree::PlainText(txt) if txt.trim().is_empty())
}

/// Perform tranformations to text. This splits paragraphs, replaces patterns,
/// and corrects newlines.
///
/// To avoid allocating strings (and using a different heavier tt type), our
/// replace method means split into three and append each. For this reason, any
/// viewer should treat consecutive `PlainText` types as belonging to the same
/// paragraph.
fn expand_plaintext<'a>(
    txt: &'a str,
    stream: &mut Vec<MdTree<'a>>,
    mut f: fn(&'a str) -> MdTree<'a>,
) {
    if txt.is_empty() {
        return;
    } else if txt == "\n" {
        if let Some(tt) = stream.last() {
            let tmp = MdTree::PlainText(" ");
            if should_break(tt, &tmp) == BreakRule::Optional {
                stream.push(tmp);
            }
        }
        return;
    }
    let mut queue1 = Vec::new();
    let mut queue2 = Vec::new();
    let stream_start_len = stream.len();
    for paragraph in txt.split("\n\n") {
        if paragraph.is_empty() {
            stream.push(MdTree::ParagraphBreak);
            continue;
        }
        let paragraph = trim_extra_ws(paragraph);

        queue1.clear();
        queue1.push(paragraph);

        for (from, to) in REPLACEMENTS {
            queue2.clear();
            for item in &queue1 {
                for s in item.split(from) {
                    queue2.extend(&[s, to]);
                }
                if queue2.len() > 1 {
                    let _ = queue2.pop(); // remove last unnecessary intersperse
                }
            }
            mem::swap(&mut queue1, &mut queue2);
        }

        // Make sure we don't double whitespace
        queue1.retain(|s| !s.is_empty());
        for idx in 0..queue1.len() {
            queue1[idx] = trim_extra_ws(queue1[idx]);
            if idx < queue1.len() - 1
                && queue1[idx].ends_with(char::is_whitespace)
                && queue1[idx + 1].starts_with(char::is_whitespace)
            {
                queue1[idx] = queue1[idx].trim_end();
            }
        }
        stream.extend(queue1.iter().copied().filter(|txt| !txt.is_empty()).map(&mut f));
        stream.push(MdTree::ParagraphBreak);
    }

    if stream.len() - stream_start_len > 1 {
        let _ = stream.pop(); // remove last unnecessary intersperse
    }
}

/// Turn reflinks (links with reference IDs) into normal standalone links using
/// listed link definitions
fn match_reflink<'a>(linkdefs: &[MdTree<'a>], disp: &'a str, match_id: Option<&str>) -> MdTree<'a> {
    let to_match = match_id.unwrap_or(disp); // Match with the display name if there isn't an id
    for def in linkdefs {
        if let MdTree::LinkDef { id, link } = def {
            if *id == to_match {
                return MdTree::Link { disp, link };
            }
        }
    }
    MdTree::Link { disp, link: "" } // link not found
}

/// If there is more than one whitespace char at start or end, trim the extras
fn trim_extra_ws(mut txt: &str) -> &str {
    let start_ws =
        txt.bytes().position(|ch| !ch.is_ascii_whitespace()).unwrap_or(txt.len()).saturating_sub(1);
    txt = &txt[start_ws..];
    let end_ws = txt
        .bytes()
        .rev()
        .position(|ch| !ch.is_ascii_whitespace())
        .unwrap_or(txt.len())
        .saturating_sub(1);
    &txt[..txt.len() - end_ws]
}

/// If there is more than one whitespace char at start, trim the extras
fn trim_ascii_start(buf: &[u8]) -> &[u8] {
    let count = buf.iter().take_while(|ch| ch.is_ascii_whitespace()).count();
    &buf[count..]
}

#[cfg(test)]
#[path = "tests/parse.rs"]
mod tests;
