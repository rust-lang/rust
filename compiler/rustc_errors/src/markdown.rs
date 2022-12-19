//! A very minimal markdown parser
//!
//! Use the entrypoint `create_ast(&str) -> MdTree` to generate the AST.
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

use regex::bytes::Regex;
use std::cmp::min;
use std::io::Error;
use std::io::Write;
use std::str::{self, from_utf8};
use std::sync::LazyLock;
use termcolor::{Buffer, BufferWriter, Color, ColorChoice, ColorSpec, WriteColor};

const NEWLINE_CHARS: &[u8; 2] = b"\r\n";
const BREAK_CHARS: &[u8; 10] = br#".,"'\;:?()"#;

/// Representation of how to match various markdown types
const PATTERNS: [MdPattern; 10] = [
    MdPattern::new(Anchor::Any("<!--"), Anchor::Any("-->"), MdType::Comment),
    MdPattern::new(Anchor::Sol("```"), Anchor::Sol("```"), MdType::CodeBlock),
    MdPattern::new(Anchor::Sol("# "), Anchor::Eol(""), MdType::Heading1),
    MdPattern::new(Anchor::Sol("## "), Anchor::Eol(""), MdType::Heading2),
    MdPattern::new(Anchor::Sol("### "), Anchor::Eol(""), MdType::Heading3),
    MdPattern::new(Anchor::Sol("#### "), Anchor::Eol(""), MdType::Heading4),
    MdPattern::new(Anchor::LeadBreak("`"), Anchor::TrailBreak("`"), MdType::CodeInline),
    MdPattern::new(Anchor::LeadBreak("**"), Anchor::TrailBreak("**"), MdType::Strong),
    MdPattern::new(Anchor::LeadBreak("_"), Anchor::TrailBreak("_"), MdType::Emphasis),
    MdPattern::new(Anchor::Sol("-"), Anchor::Eol(""), MdType::ListItem),
    // MdPattern::new(Anchor::Any("\n\n"),Anchor::Any(""))
    // strikethrough
];

/// This is an example for using doc comment attributes
static RE_URL_NAMED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
    r"(?P<pre_ws>^|\s|[[:punct:]])\[(?P<display>.+)\]\((?P<url>\S+)\)(?P<post_ws>$|\s|[[:punct:]])"
).unwrap()
});
static RE_URL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?P<pre_ws>^|\s|[[:punct:]])<(?P<url>\S+)>(?P<post_ws>$|\s|[[:punct:]])").unwrap()
});

/// An AST representation of a Markdown document
#[derive(Debug, PartialEq, Clone)]
pub enum MdTree<'a> {
    /// Leaf types
    Comment(&'a str),
    CodeBlock(&'a str),
    CodeInline(&'a str),
    Strong(&'a str),
    Emphasis(&'a str),
    PlainText(&'a str),
    /// Nexting types
    Heading1(Vec<MdTree<'a>>),
    Heading2(Vec<MdTree<'a>>),
    Heading3(Vec<MdTree<'a>>),
    Heading4(Vec<MdTree<'a>>),
    ListItem(Vec<MdTree<'a>>),
    /// Root node
    Root(Vec<MdTree<'a>>),
}

impl<'a> MdTree<'a> {
    /// Create a `MdTree` from a string and a specified type
    fn from_type(s: &'a str, tag: MdType) -> Self {
        match tag {
            MdType::Comment => Self::Comment(s),
            MdType::CodeBlock => Self::CodeBlock(s),
            MdType::CodeInline => Self::CodeInline(s),
            MdType::Strong => Self::Strong(s),
            MdType::Emphasis => Self::Emphasis(s),
            MdType::Heading1 => Self::Heading1(vec![MdTree::PlainText(s)]),
            MdType::Heading2 => Self::Heading2(vec![MdTree::PlainText(s)]),
            MdType::Heading3 => Self::Heading3(vec![MdTree::PlainText(s)]),
            MdType::Heading4 => Self::Heading4(vec![MdTree::PlainText(s)]),
            MdType::ListItem => Self::ListItem(vec![MdTree::PlainText(s)]),
        }
    }

    /// Print to terminal output, resetting to the default style after each
    fn term_print_vec(
        v: &Vec<Self>,
        buf: &mut Buffer,
        default: Option<&ColorSpec>,
        is_firstline: &mut bool,
    ) -> Result<(), Error> {
        match default {
            Some(c) => buf.set_color(c)?,
            None => buf.reset()?,
        }

        for item in v {
            item.term_write_recurse(buf, is_firstline)?;
            if let Some(c) = default {
                buf.set_color(c)?;
            }
        }

        buf.reset()?;
        Ok(())
    }

    fn term_write_recurse(&self, buf: &mut Buffer, is_firstline: &mut bool) -> Result<(), Error> {
        match self {
            // Do nothing
            MdTree::Comment(_) => (),
            MdTree::CodeBlock(s) => {
                buf.set_color(ColorSpec::new().set_dimmed(true))?;
                write_if_not_first(buf, "\n", is_firstline)?;
                // Account for "```rust\n..." starts of strings
                if !s.starts_with('\n') {
                    write!(buf, "{}\n", s.split('\n').nth(1).unwrap_or(s).trim())?;
                } else {
                    write!(buf, "{}\n", s.trim())?;
                }
            }
            MdTree::CodeInline(s) => {
                buf.set_color(ColorSpec::new().set_dimmed(true))?;
                write!(buf, "{s}")?;
            }
            MdTree::Strong(s) => {
                buf.set_color(ColorSpec::new().set_bold(true))?;
                buf.write(&write_replace(s))?;
            }
            MdTree::Emphasis(s) => {
                buf.set_color(ColorSpec::new().set_italic(true))?;
                buf.write(&write_replace(s))?;
            }
            MdTree::Heading1(v) => {
                write_if_not_first(buf, "\n", is_firstline)?;
                Self::term_print_vec(
                    v,
                    buf,
                    Some(
                        ColorSpec::new()
                            .set_fg(Some(Color::Cyan))
                            .set_intense(true)
                            .set_bold(true)
                            .set_underline(true),
                    ),
                    is_firstline,
                )?;
            }
            MdTree::Heading2(v) => {
                write_if_not_first(buf, "\n", is_firstline)?;
                Self::term_print_vec(
                    v,
                    buf,
                    Some(
                        ColorSpec::new()
                            .set_fg(Some(Color::Cyan))
                            .set_intense(true)
                            .set_underline(true),
                    ),
                    is_firstline,
                )?;
            }
            MdTree::Heading3(v) => {
                Self::term_print_vec(
                    v,
                    buf,
                    Some(
                        ColorSpec::new()
                            .set_fg(Some(Color::Cyan))
                            .set_intense(true)
                            .set_italic(true),
                    ),
                    is_firstline,
                )?;
            }
            MdTree::Heading4(v) => {
                Self::term_print_vec(
                    v,
                    buf,
                    Some(
                        ColorSpec::new()
                            .set_fg(Some(Color::Cyan))
                            .set_underline(true)
                            .set_italic(true),
                    ),
                    is_firstline,
                )?;
            }
            MdTree::ListItem(v) => {
                write!(buf, "* ")?;
                Self::term_print_vec(v, buf, None, is_firstline)?;
            }
            MdTree::Root(v) => Self::term_print_vec(v, buf, None, is_firstline)?,
            MdTree::PlainText(s) => {
                buf.write(&write_replace(s))?;
            }
        }

        buf.reset()?;
        Ok(())
    }

    pub fn write_termcolor_buf(&self, buf: &mut Buffer) {
        let _ = self.term_write_recurse(buf, &mut true);
    }
}

/// Grumble grumble workaround for not being able to `as` cast mixed field enums
#[derive(Debug, PartialEq, Copy, Clone)]
enum MdType {
    Comment,
    CodeBlock,
    CodeInline,
    Heading1,
    Heading2,
    Heading3,
    Heading4,
    Strong,
    Emphasis,
    ListItem,
}

/// A representation of the requirements to match a pattern
#[derive(Debug, PartialEq, Clone)]
enum Anchor {
    /// Start of line
    Sol(&'static str),
    /// End of line
    Eol(&'static str),
    /// Preceded by whitespace or punctuation
    LeadBreak(&'static str),
    /// Precedes whitespace or punctuation
    TrailBreak(&'static str),
    /// Plain pattern matching
    Any(&'static str),
}

impl Anchor {
    /// Get any inner value
    const fn unwrap(&self) -> &str {
        match self {
            Self::Sol(s)
            | Self::Eol(s)
            | Self::LeadBreak(s)
            | Self::TrailBreak(s)
            | Self::Any(s) => s,
        }
    }
}

/// Context used for pattern matching
#[derive(Debug, PartialEq, Clone)]
struct Context {
    at_line_start: bool,
    preceded_by_break: bool,
}

/// A simple markdown type
#[derive(Debug, PartialEq, Clone)]
struct MdPattern {
    start: Anchor,
    end: Anchor,
    tag: MdType,
}

/// Return matched data and leftover data
#[derive(Debug, PartialEq, Clone)]
struct MdResult<'a> {
    matched: MdTree<'a>,
    residual: &'a [u8],
}

impl MdPattern {
    const fn new(start: Anchor, end: Anchor, tag: MdType) -> Self {
        Self { start, end, tag }
    }

    /// Given a string like `match]residual`, return `match` and `residual` within `MdResult`
    fn parse_end<'a>(&self, bytes: &'a [u8], ctx: &Context) -> MdResult<'a> {
        let mut i = 0usize;
        let mut at_line_start: bool; // whether this index is the start of a line
        let mut next_at_line_start = ctx.at_line_start;
        let anchor = &self.end;
        let pat_end = anchor.unwrap().as_bytes();

        while i < bytes.len() {
            let working = &bytes[i..];
            at_line_start = next_at_line_start;
            next_at_line_start = NEWLINE_CHARS.contains(&working[0]);

            if !working.starts_with(pat_end) {
                i += 1;
                continue;
            }

            // Our pattern matches. Just break if there is no remaining
            // string
            let residual = &working[pat_end.len()..];

            let Some(next_byte) = residual.first()  else {
                break
            };

            // Validate postconditions if we have a remaining string
            let is_matched = match anchor {
                Anchor::TrailBreak(_) => is_break_char(*next_byte),
                Anchor::Eol(_) => NEWLINE_CHARS.contains(next_byte),
                Anchor::Sol(_) => at_line_start,
                Anchor::Any(_) => true,
                Anchor::LeadBreak(_) => panic!("unexpected end pattern"),
            };

            if is_matched {
                break;
            }
            i += 1;
        }

        let matched = MdTree::from_type(from_utf8(&bytes[..i]).unwrap(), self.tag);
        let residual = &bytes[min(bytes.len(), i + pat_end.len())..];

        MdResult { matched, residual }
    }

    /// Given a string like `[match]residual`, return `MdTree(match)` and `residual`
    ///
    /// Return `None` if the string does not start with the correct pattern
    fn parse_start<'a>(&self, bytes: &'a [u8], ctx: &Context) -> Option<MdResult<'a>> {
        // Guard for strings that do not match
        if !ctx.at_line_start && matches!(self.start, Anchor::Sol(_)) {
            return None;
        }
        if !ctx.preceded_by_break && matches!(self.start, Anchor::LeadBreak(_)) {
            return None;
        }

        // Return if we don't start with the pattern
        let pat_start = self.start.unwrap().as_bytes();
        if !bytes.starts_with(pat_start) {
            return None;
        }

        // We have a match, parse to the closing delimiter
        let residual = &bytes[pat_start.len()..];
        Some(self.parse_end(residual, ctx))
    }
}

/// Apply `recurse_tree` to each element in a vector
fn recurse_vec<'a>(v: Vec<MdTree<'a>>) -> Vec<MdTree<'a>> {
    v.into_iter().flat_map(recurse_tree).collect()
}

/// Given a `MdTree`, expand all children
fn recurse_tree<'a>(tree: MdTree<'a>) -> Vec<MdTree<'a>> {
    match tree {
        // Leaf nodes; just add
        MdTree::Comment(_)
        | MdTree::CodeBlock(_)
        | MdTree::CodeInline(_)
        | MdTree::Strong(_)
        | MdTree::Emphasis(_) => vec![tree],
        // Leaf node with possible further expansion
        MdTree::PlainText(s) => parse_str(s),
        // Parent nodes; recurse these and add
        MdTree::Heading1(v) => vec![MdTree::Heading1(recurse_vec(v))],
        MdTree::Heading2(v) => vec![MdTree::Heading2(recurse_vec(v))],
        MdTree::Heading3(v) => vec![MdTree::Heading3(recurse_vec(v))],
        MdTree::Heading4(v) => vec![MdTree::Heading4(recurse_vec(v))],
        MdTree::ListItem(v) => vec![MdTree::ListItem(recurse_vec(v))],
        MdTree::Root(v) => vec![MdTree::Root(recurse_vec(v))],
    }
}

/// Main parser function for a single string
fn parse_str<'a>(s: &'a str) -> Vec<MdTree<'a>> {
    let mut v: Vec<MdTree<'_>> = Vec::new();
    let mut ctx = Context { at_line_start: true, preceded_by_break: true };
    let mut next_ctx = ctx.clone();
    let mut working = s.as_bytes();
    let mut i = 0;

    while i < working.len() {
        let test_slice = &working[i..];
        let current_char = test_slice.first().unwrap();

        ctx = next_ctx.clone();
        next_ctx.at_line_start = NEWLINE_CHARS.contains(current_char);
        next_ctx.preceded_by_break = is_break_char(*current_char);

        let found = PATTERNS.iter().find_map(|p| p.parse_start(&working[i..], &ctx));

        let Some(res) = found else {
            i += 1;
            continue;
        };

        if i > 0 {
            v.push(MdTree::PlainText(from_utf8(&working[..i]).unwrap()));
        }
        v.append(&mut recurse_tree(res.matched));
        working = res.residual;
        i = 0;
    }

    if i > 0 {
        v.push(MdTree::PlainText(from_utf8(&working[..i]).unwrap()));
    }

    v
}

/// Test if a character is whitespace or a breaking character (punctuation)
fn is_break_char(c: u8) -> bool {
    c.is_ascii_whitespace() || BREAK_CHARS.contains(&c)
}

#[must_use]
pub fn create_ast<'a>(s: &'a str) -> MdTree<'a> {
    MdTree::Root(parse_str(s))
}

fn write_replace(s: &str) -> Vec<u8> {
    const REPLACEMENTS: [(&str, &str); 7] = [
        ("(c)", "©"),
        ("(C)", "©"),
        ("(r)", "®"),
        ("(R)", "®"),
        ("(tm)", "™"),
        ("(TM)", "™"),
        (":crab:", "🦀"),
    ];

    let mut ret = s.to_owned();
    for (from, to) in REPLACEMENTS {
        ret = ret.replace(from, to);
    }

    let tmp = RE_URL_NAMED.replace_all(
        ret.as_bytes(),
        b"${pre_ws}\x1b]8;;${url}\x1b\\${display}\x1b]8;;\x1b\\${post_ws}".as_slice(),
    );
    RE_URL
        .replace_all(
            &tmp,
            b"${pre_ws}\x1b]8;;${url}\x1b\\${url}\x1b]8;;\x1b\\${post_ws}".as_slice(),
        )
        .to_vec()
}

/// Write something to the buf if some first indicator is false, then set the indicator false
fn write_if_not_first(buf: &mut Buffer, s: &str, is_firstline: &mut bool) -> Result<(), Error> {
    if *is_firstline {
        *is_firstline = false;
    } else {
        write!(buf, "{s}")?;
    }
    Ok(())
}

pub fn create_stdout_bufwtr() -> BufferWriter {
    BufferWriter::stdout(ColorChoice::Always)
}

#[cfg(test)]
mod tests;
