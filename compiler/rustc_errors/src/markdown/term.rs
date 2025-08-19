use std::cell::Cell;
use std::io::{self, Write};

use termcolor::{Buffer, Color, ColorSpec, WriteColor};

use crate::markdown::{MdStream, MdTree};

const DEFAULT_COLUMN_WIDTH: usize = 140;

thread_local! {
    /// Track the position of viewable characters in our buffer
    static CURSOR: Cell<usize> = const { Cell::new(0) };
    /// Width of the terminal
    static WIDTH: Cell<usize> = const { Cell::new(DEFAULT_COLUMN_WIDTH) };
}

/// Print to terminal output to a buffer
pub(crate) fn entrypoint(stream: &MdStream<'_>, buf: &mut Buffer) -> io::Result<()> {
    #[cfg(not(test))]
    if let Some((w, _)) = termize::dimensions() {
        WIDTH.set(std::cmp::min(w, DEFAULT_COLUMN_WIDTH));
    }
    write_stream(stream, buf, None, 0)?;
    buf.write_all(b"\n")
}

/// Write the buffer, reset to the default style after each
fn write_stream(
    MdStream(stream): &MdStream<'_>,
    buf: &mut Buffer,
    default: Option<&ColorSpec>,
    indent: usize,
) -> io::Result<()> {
    match default {
        Some(c) => buf.set_color(c)?,
        None => buf.reset()?,
    }

    for tt in stream {
        write_tt(tt, buf, indent)?;
        if let Some(c) = default {
            buf.set_color(c)?;
        }
    }

    buf.reset()?;
    Ok(())
}

fn write_tt(tt: &MdTree<'_>, buf: &mut Buffer, indent: usize) -> io::Result<()> {
    match tt {
        MdTree::CodeBlock { txt, lang: _ } => {
            buf.set_color(ColorSpec::new().set_dimmed(true))?;
            buf.write_all(txt.as_bytes())?;
        }
        MdTree::CodeInline(txt) => {
            buf.set_color(ColorSpec::new().set_dimmed(true))?;
            write_wrapping(buf, txt, indent, None)?;
        }
        MdTree::Strong(txt) => {
            buf.set_color(ColorSpec::new().set_bold(true))?;
            write_wrapping(buf, txt, indent, None)?;
        }
        MdTree::Emphasis(txt) => {
            buf.set_color(ColorSpec::new().set_italic(true))?;
            write_wrapping(buf, txt, indent, None)?;
        }
        MdTree::Strikethrough(txt) => {
            buf.set_color(ColorSpec::new().set_strikethrough(true))?;
            write_wrapping(buf, txt, indent, None)?;
        }
        MdTree::PlainText(txt) => {
            write_wrapping(buf, txt, indent, None)?;
        }
        MdTree::Link { disp, link } => {
            write_wrapping(buf, disp, indent, Some(link))?;
        }
        MdTree::ParagraphBreak => {
            buf.write_all(b"\n\n")?;
            reset_cursor();
        }
        MdTree::LineBreak => {
            buf.write_all(b"\n")?;
            reset_cursor();
        }
        MdTree::HorizontalRule => {
            (0..WIDTH.get()).for_each(|_| buf.write_all(b"-").unwrap());
            reset_cursor();
        }
        MdTree::Heading(n, stream) => {
            let mut cs = ColorSpec::new();
            cs.set_fg(Some(Color::Cyan));
            match n {
                1 => cs.set_intense(true).set_bold(true).set_underline(true),
                2 => cs.set_intense(true).set_underline(true),
                3 => cs.set_intense(true).set_italic(true),
                4.. => cs.set_underline(true).set_italic(true),
                0 => unreachable!(),
            };
            write_stream(stream, buf, Some(&cs), 0)?;
            buf.write_all(b"\n")?;
        }
        MdTree::OrderedListItem(n, stream) => {
            let base = format!("{n}. ");
            write_wrapping(buf, &format!("{base:<4}"), indent, None)?;
            write_stream(stream, buf, None, indent + 4)?;
        }
        MdTree::UnorderedListItem(stream) => {
            let base = "* ";
            write_wrapping(buf, &format!("{base:<4}"), indent, None)?;
            write_stream(stream, buf, None, indent + 4)?;
        }
        // Patterns popped in previous step
        MdTree::Comment(_) | MdTree::LinkDef { .. } | MdTree::RefLink { .. } => unreachable!(),
    }

    buf.reset()?;

    Ok(())
}

/// End of that block, just wrap the line
fn reset_cursor() {
    CURSOR.set(0);
}

/// Change to be generic on Write for testing. If we have a link URL, we don't
/// count the extra tokens to make it clickable.
fn write_wrapping<B: io::Write>(
    buf: &mut B,
    text: &str,
    indent: usize,
    link_url: Option<&str>,
) -> io::Result<()> {
    let ind_ws = &b"          "[..indent];
    let mut to_write = text;
    if let Some(url) = link_url {
        // This is a nonprinting prefix so we don't increment our cursor
        write!(buf, "\x1b]8;;{url}\x1b\\")?;
    }
    CURSOR.with(|cur| {
        loop {
            if cur.get() == 0 {
                buf.write_all(ind_ws)?;
                cur.set(indent);
            }
            let ch_count = WIDTH.get() - cur.get();
            let mut iter = to_write.char_indices();
            let Some((end_idx, _ch)) = iter.nth(ch_count) else {
                // Write entire line
                buf.write_all(to_write.as_bytes())?;
                cur.set(cur.get() + to_write.chars().count());
                break;
            };

            if let Some((break_idx, ch)) = to_write[..end_idx]
                .char_indices()
                .rev()
                .find(|(_idx, ch)| ch.is_whitespace() || ['_', '-'].contains(ch))
            {
                // Found whitespace to break at
                if ch.is_whitespace() {
                    writeln!(buf, "{}", &to_write[..break_idx])?;
                    to_write = to_write[break_idx..].trim_start();
                } else {
                    // Break at a `-` or `_` separator
                    writeln!(buf, "{}", &to_write.get(..break_idx + 1).unwrap_or(to_write))?;
                    to_write = to_write.get(break_idx + 1..).unwrap_or_default().trim_start();
                }
            } else {
                // No whitespace, we need to just split
                let ws_idx =
                    iter.find(|(_, ch)| ch.is_whitespace()).map_or(to_write.len(), |(idx, _)| idx);
                writeln!(buf, "{}", &to_write[..ws_idx])?;
                to_write = to_write.get(ws_idx + 1..).map_or("", str::trim_start);
            }
            cur.set(0);
        }
        if link_url.is_some() {
            buf.write_all(b"\x1b]8;;\x1b\\")?;
        }

        Ok(())
    })
}

#[cfg(test)]
#[path = "tests/term.rs"]
mod tests;
