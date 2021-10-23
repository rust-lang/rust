// Formatting and tools for comments.

use std::{self, borrow::Cow, iter};

use itertools::{multipeek, MultiPeek};
use rustc_span::Span;

use crate::config::Config;
use crate::rewrite::RewriteContext;
use crate::shape::{Indent, Shape};
use crate::string::{rewrite_string, StringFormat};
use crate::utils::{
    count_newlines, first_line_width, last_line_width, trim_left_preserve_layout,
    trimmed_last_line_width, unicode_str_width,
};
use crate::{ErrorKind, FormattingError};

fn is_custom_comment(comment: &str) -> bool {
    if !comment.starts_with("//") {
        false
    } else if let Some(c) = comment.chars().nth(2) {
        !c.is_alphanumeric() && !c.is_whitespace()
    } else {
        false
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum CommentStyle<'a> {
    DoubleSlash,
    TripleSlash,
    Doc,
    SingleBullet,
    DoubleBullet,
    Exclamation,
    Custom(&'a str),
}

fn custom_opener(s: &str) -> &str {
    s.lines().next().map_or("", |first_line| {
        first_line
            .find(' ')
            .map_or(first_line, |space_index| &first_line[0..=space_index])
    })
}

impl<'a> CommentStyle<'a> {
    /// Returns `true` if the commenting style covers a line only.
    pub(crate) fn is_line_comment(&self) -> bool {
        match *self {
            CommentStyle::DoubleSlash
            | CommentStyle::TripleSlash
            | CommentStyle::Doc
            | CommentStyle::Custom(_) => true,
            _ => false,
        }
    }

    /// Returns `true` if the commenting style can span over multiple lines.
    pub(crate) fn is_block_comment(&self) -> bool {
        match *self {
            CommentStyle::SingleBullet | CommentStyle::DoubleBullet | CommentStyle::Exclamation => {
                true
            }
            _ => false,
        }
    }

    /// Returns `true` if the commenting style is for documentation.
    pub(crate) fn is_doc_comment(&self) -> bool {
        matches!(*self, CommentStyle::TripleSlash | CommentStyle::Doc)
    }

    pub(crate) fn opener(&self) -> &'a str {
        match *self {
            CommentStyle::DoubleSlash => "// ",
            CommentStyle::TripleSlash => "/// ",
            CommentStyle::Doc => "//! ",
            CommentStyle::SingleBullet => "/* ",
            CommentStyle::DoubleBullet => "/** ",
            CommentStyle::Exclamation => "/*! ",
            CommentStyle::Custom(opener) => opener,
        }
    }

    pub(crate) fn closer(&self) -> &'a str {
        match *self {
            CommentStyle::DoubleSlash
            | CommentStyle::TripleSlash
            | CommentStyle::Custom(..)
            | CommentStyle::Doc => "",
            CommentStyle::SingleBullet | CommentStyle::DoubleBullet | CommentStyle::Exclamation => {
                " */"
            }
        }
    }

    pub(crate) fn line_start(&self) -> &'a str {
        match *self {
            CommentStyle::DoubleSlash => "// ",
            CommentStyle::TripleSlash => "/// ",
            CommentStyle::Doc => "//! ",
            CommentStyle::SingleBullet | CommentStyle::DoubleBullet | CommentStyle::Exclamation => {
                " * "
            }
            CommentStyle::Custom(opener) => opener,
        }
    }

    pub(crate) fn to_str_tuplet(&self) -> (&'a str, &'a str, &'a str) {
        (self.opener(), self.closer(), self.line_start())
    }
}

pub(crate) fn comment_style(orig: &str, normalize_comments: bool) -> CommentStyle<'_> {
    if !normalize_comments {
        if orig.starts_with("/**") && !orig.starts_with("/**/") {
            CommentStyle::DoubleBullet
        } else if orig.starts_with("/*!") {
            CommentStyle::Exclamation
        } else if orig.starts_with("/*") {
            CommentStyle::SingleBullet
        } else if orig.starts_with("///") && orig.chars().nth(3).map_or(true, |c| c != '/') {
            CommentStyle::TripleSlash
        } else if orig.starts_with("//!") {
            CommentStyle::Doc
        } else if is_custom_comment(orig) {
            CommentStyle::Custom(custom_opener(orig))
        } else {
            CommentStyle::DoubleSlash
        }
    } else if (orig.starts_with("///") && orig.chars().nth(3).map_or(true, |c| c != '/'))
        || (orig.starts_with("/**") && !orig.starts_with("/**/"))
    {
        CommentStyle::TripleSlash
    } else if orig.starts_with("//!") || orig.starts_with("/*!") {
        CommentStyle::Doc
    } else if is_custom_comment(orig) {
        CommentStyle::Custom(custom_opener(orig))
    } else {
        CommentStyle::DoubleSlash
    }
}

/// Returns true if the last line of the passed string finishes with a block-comment.
pub(crate) fn is_last_comment_block(s: &str) -> bool {
    s.trim_end().ends_with("*/")
}

/// Combine `prev_str` and `next_str` into a single `String`. `span` may contain
/// comments between two strings. If there are such comments, then that will be
/// recovered. If `allow_extend` is true and there is no comment between the two
/// strings, then they will be put on a single line as long as doing so does not
/// exceed max width.
pub(crate) fn combine_strs_with_missing_comments(
    context: &RewriteContext<'_>,
    prev_str: &str,
    next_str: &str,
    span: Span,
    shape: Shape,
    allow_extend: bool,
) -> Option<String> {
    trace!(
        "combine_strs_with_missing_comments `{}` `{}` {:?} {:?}",
        prev_str,
        next_str,
        span,
        shape
    );

    let mut result =
        String::with_capacity(prev_str.len() + next_str.len() + shape.indent.width() + 128);
    result.push_str(prev_str);
    let mut allow_one_line = !prev_str.contains('\n') && !next_str.contains('\n');
    let first_sep =
        if prev_str.is_empty() || next_str.is_empty() || trimmed_last_line_width(prev_str) == 0 {
            ""
        } else {
            " "
        };
    let mut one_line_width =
        last_line_width(prev_str) + first_line_width(next_str) + first_sep.len();

    let config = context.config;
    let indent = shape.indent;
    let missing_comment = rewrite_missing_comment(span, shape, context)?;

    if missing_comment.is_empty() {
        if allow_extend && one_line_width <= shape.width {
            result.push_str(first_sep);
        } else if !prev_str.is_empty() {
            result.push_str(&indent.to_string_with_newline(config))
        }
        result.push_str(next_str);
        return Some(result);
    }

    // We have a missing comment between the first expression and the second expression.

    // Peek the the original source code and find out whether there is a newline between the first
    // expression and the second expression or the missing comment. We will preserve the original
    // layout whenever possible.
    let original_snippet = context.snippet(span);
    let prefer_same_line = if let Some(pos) = original_snippet.find('/') {
        !original_snippet[..pos].contains('\n')
    } else {
        !original_snippet.contains('\n')
    };

    one_line_width -= first_sep.len();
    let first_sep = if prev_str.is_empty() || missing_comment.is_empty() {
        Cow::from("")
    } else {
        let one_line_width = last_line_width(prev_str) + first_line_width(&missing_comment) + 1;
        if prefer_same_line && one_line_width <= shape.width {
            Cow::from(" ")
        } else {
            indent.to_string_with_newline(config)
        }
    };
    result.push_str(&first_sep);
    result.push_str(&missing_comment);

    let second_sep = if missing_comment.is_empty() || next_str.is_empty() {
        Cow::from("")
    } else if missing_comment.starts_with("//") {
        indent.to_string_with_newline(config)
    } else {
        one_line_width += missing_comment.len() + first_sep.len() + 1;
        allow_one_line &= !missing_comment.starts_with("//") && !missing_comment.contains('\n');
        if prefer_same_line && allow_one_line && one_line_width <= shape.width {
            Cow::from(" ")
        } else {
            indent.to_string_with_newline(config)
        }
    };
    result.push_str(&second_sep);
    result.push_str(next_str);

    Some(result)
}

pub(crate) fn rewrite_doc_comment(orig: &str, shape: Shape, config: &Config) -> Option<String> {
    identify_comment(orig, false, shape, config, true)
}

pub(crate) fn rewrite_comment(
    orig: &str,
    block_style: bool,
    shape: Shape,
    config: &Config,
) -> Option<String> {
    identify_comment(orig, block_style, shape, config, false)
}

fn identify_comment(
    orig: &str,
    block_style: bool,
    shape: Shape,
    config: &Config,
    is_doc_comment: bool,
) -> Option<String> {
    let style = comment_style(orig, false);

    // Computes the byte length of line taking into account a newline if the line is part of a
    // paragraph.
    fn compute_len(orig: &str, line: &str) -> usize {
        if orig.len() > line.len() {
            if orig.as_bytes()[line.len()] == b'\r' {
                line.len() + 2
            } else {
                line.len() + 1
            }
        } else {
            line.len()
        }
    }

    // Get the first group of line comments having the same commenting style.
    //
    // Returns a tuple with:
    // - a boolean indicating if there is a blank line
    // - a number indicating the size of the first group of comments
    fn consume_same_line_comments(
        style: CommentStyle<'_>,
        orig: &str,
        line_start: &str,
    ) -> (bool, usize) {
        let mut first_group_ending = 0;
        let mut hbl = false;

        for line in orig.lines() {
            let trimmed_line = line.trim_start();
            if trimmed_line.is_empty() {
                hbl = true;
                break;
            } else if trimmed_line.starts_with(line_start)
                || comment_style(trimmed_line, false) == style
            {
                first_group_ending += compute_len(&orig[first_group_ending..], line);
            } else {
                break;
            }
        }
        (hbl, first_group_ending)
    }

    let (has_bare_lines, first_group_ending) = match style {
        CommentStyle::DoubleSlash | CommentStyle::TripleSlash | CommentStyle::Doc => {
            let line_start = style.line_start().trim_start();
            consume_same_line_comments(style, orig, line_start)
        }
        CommentStyle::Custom(opener) => {
            let trimmed_opener = opener.trim_end();
            consume_same_line_comments(style, orig, trimmed_opener)
        }
        // for a block comment, search for the closing symbol
        CommentStyle::DoubleBullet | CommentStyle::SingleBullet | CommentStyle::Exclamation => {
            let closer = style.closer().trim_start();
            let mut count = orig.matches(closer).count();
            let mut closing_symbol_offset = 0;
            let mut hbl = false;
            let mut first = true;
            for line in orig.lines() {
                closing_symbol_offset += compute_len(&orig[closing_symbol_offset..], line);
                let mut trimmed_line = line.trim_start();
                if !trimmed_line.starts_with('*')
                    && !trimmed_line.starts_with("//")
                    && !trimmed_line.starts_with("/*")
                {
                    hbl = true;
                }

                // Remove opener from consideration when searching for closer
                if first {
                    let opener = style.opener().trim_end();
                    trimmed_line = &trimmed_line[opener.len()..];
                    first = false;
                }
                if trimmed_line.ends_with(closer) {
                    count -= 1;
                    if count == 0 {
                        break;
                    }
                }
            }
            (hbl, closing_symbol_offset)
        }
    };

    let (first_group, rest) = orig.split_at(first_group_ending);
    let rewritten_first_group =
        if !config.normalize_comments() && has_bare_lines && style.is_block_comment() {
            trim_left_preserve_layout(first_group, shape.indent, config)?
        } else if !config.normalize_comments()
            && !config.wrap_comments()
            && !config.format_code_in_doc_comments()
        {
            light_rewrite_comment(first_group, shape.indent, config, is_doc_comment)
        } else {
            rewrite_comment_inner(
                first_group,
                block_style,
                style,
                shape,
                config,
                is_doc_comment || style.is_doc_comment(),
            )?
        };
    if rest.is_empty() {
        Some(rewritten_first_group)
    } else {
        identify_comment(
            rest.trim_start(),
            block_style,
            shape,
            config,
            is_doc_comment,
        )
        .map(|rest_str| {
            format!(
                "{}\n{}{}{}",
                rewritten_first_group,
                // insert back the blank line
                if has_bare_lines && style.is_line_comment() {
                    "\n"
                } else {
                    ""
                },
                shape.indent.to_string(config),
                rest_str
            )
        })
    }
}

/// Enum indicating if the code block contains rust based on attributes
enum CodeBlockAttribute {
    Rust,
    NotRust,
}

impl CodeBlockAttribute {
    /// Parse comma separated attributes list. Return rust only if all
    /// attributes are valid rust attributes
    /// See <https://doc.rust-lang.org/rustdoc/print.html#attributes>
    fn new(attributes: &str) -> CodeBlockAttribute {
        for attribute in attributes.split(',') {
            match attribute.trim() {
                "" | "rust" | "should_panic" | "no_run" | "edition2015" | "edition2018"
                | "edition2021" => (),
                "ignore" | "compile_fail" | "text" => return CodeBlockAttribute::NotRust,
                _ => return CodeBlockAttribute::NotRust,
            }
        }
        CodeBlockAttribute::Rust
    }
}

/// Block that is formatted as an item.
///
/// An item starts with either a star `*` or a dash `-`. Different level of indentation are
/// handled by shrinking the shape accordingly.
struct ItemizedBlock {
    /// the lines that are identified as part of an itemized block
    lines: Vec<String>,
    /// the number of whitespaces up to the item sigil
    indent: usize,
    /// the string that marks the start of an item
    opener: String,
    /// sequence of whitespaces to prefix new lines that are part of the item
    line_start: String,
}

impl ItemizedBlock {
    /// Returns `true` if the line is formatted as an item
    fn is_itemized_line(line: &str) -> bool {
        let trimmed = line.trim_start();
        trimmed.starts_with("* ") || trimmed.starts_with("- ")
    }

    /// Creates a new ItemizedBlock described with the given line.
    /// The `is_itemized_line` needs to be called first.
    fn new(line: &str) -> ItemizedBlock {
        let space_to_sigil = line.chars().take_while(|c| c.is_whitespace()).count();
        let indent = space_to_sigil + 2;
        ItemizedBlock {
            lines: vec![line[indent..].to_string()],
            indent,
            opener: line[..indent].to_string(),
            line_start: " ".repeat(indent),
        }
    }

    /// Returns a `StringFormat` used for formatting the content of an item.
    fn create_string_format<'a>(&'a self, fmt: &'a StringFormat<'_>) -> StringFormat<'a> {
        StringFormat {
            opener: "",
            closer: "",
            line_start: "",
            line_end: "",
            shape: Shape::legacy(fmt.shape.width.saturating_sub(self.indent), Indent::empty()),
            trim_end: true,
            config: fmt.config,
        }
    }

    /// Returns `true` if the line is part of the current itemized block.
    /// If it is, then it is added to the internal lines list.
    fn add_line(&mut self, line: &str) -> bool {
        if !ItemizedBlock::is_itemized_line(line)
            && self.indent <= line.chars().take_while(|c| c.is_whitespace()).count()
        {
            self.lines.push(line.to_string());
            return true;
        }
        false
    }

    /// Returns the block as a string, with each line trimmed at the start.
    fn trimmed_block_as_string(&self) -> String {
        self.lines
            .iter()
            .map(|line| format!("{} ", line.trim_start()))
            .collect::<String>()
    }

    /// Returns the block as a string under its original form.
    fn original_block_as_string(&self) -> String {
        self.lines.join("\n")
    }
}

struct CommentRewrite<'a> {
    result: String,
    code_block_buffer: String,
    is_prev_line_multi_line: bool,
    code_block_attr: Option<CodeBlockAttribute>,
    item_block: Option<ItemizedBlock>,
    comment_line_separator: String,
    indent_str: String,
    max_width: usize,
    fmt_indent: Indent,
    fmt: StringFormat<'a>,

    opener: String,
    closer: String,
    line_start: String,
}

impl<'a> CommentRewrite<'a> {
    fn new(
        orig: &'a str,
        block_style: bool,
        shape: Shape,
        config: &'a Config,
    ) -> CommentRewrite<'a> {
        let (opener, closer, line_start) = if block_style {
            CommentStyle::SingleBullet.to_str_tuplet()
        } else {
            comment_style(orig, config.normalize_comments()).to_str_tuplet()
        };

        let max_width = shape
            .width
            .checked_sub(closer.len() + opener.len())
            .unwrap_or(1);
        let indent_str = shape.indent.to_string_with_newline(config).to_string();

        let mut cr = CommentRewrite {
            result: String::with_capacity(orig.len() * 2),
            code_block_buffer: String::with_capacity(128),
            is_prev_line_multi_line: false,
            code_block_attr: None,
            item_block: None,
            comment_line_separator: format!("{}{}", indent_str, line_start),
            max_width,
            indent_str,
            fmt_indent: shape.indent,

            fmt: StringFormat {
                opener: "",
                closer: "",
                line_start,
                line_end: "",
                shape: Shape::legacy(max_width, shape.indent),
                trim_end: true,
                config,
            },

            opener: opener.to_owned(),
            closer: closer.to_owned(),
            line_start: line_start.to_owned(),
        };
        cr.result.push_str(opener);
        cr
    }

    fn join_block(s: &str, sep: &str) -> String {
        let mut result = String::with_capacity(s.len() + 128);
        let mut iter = s.lines().peekable();
        while let Some(line) = iter.next() {
            result.push_str(line);
            result.push_str(match iter.peek() {
                Some(next_line) if next_line.is_empty() => sep.trim_end(),
                Some(..) => sep,
                None => "",
            });
        }
        result
    }

    fn finish(mut self) -> String {
        if !self.code_block_buffer.is_empty() {
            // There is a code block that is not properly enclosed by backticks.
            // We will leave them untouched.
            self.result.push_str(&self.comment_line_separator);
            self.result.push_str(&Self::join_block(
                &trim_custom_comment_prefix(&self.code_block_buffer),
                &self.comment_line_separator,
            ));
        }

        if let Some(ref ib) = self.item_block {
            // the last few lines are part of an itemized block
            self.fmt.shape = Shape::legacy(self.max_width, self.fmt_indent);
            let item_fmt = ib.create_string_format(&self.fmt);
            self.result.push_str(&self.comment_line_separator);
            self.result.push_str(&ib.opener);
            match rewrite_string(
                &ib.trimmed_block_as_string(),
                &item_fmt,
                self.max_width.saturating_sub(ib.indent),
            ) {
                Some(s) => self.result.push_str(&Self::join_block(
                    &s,
                    &format!("{}{}", self.comment_line_separator, ib.line_start),
                )),
                None => self.result.push_str(&Self::join_block(
                    &ib.original_block_as_string(),
                    &self.comment_line_separator,
                )),
            };
        }

        self.result.push_str(&self.closer);
        if self.result.ends_with(&self.opener) && self.opener.ends_with(' ') {
            // Trailing space.
            self.result.pop();
        }

        self.result
    }

    fn handle_line(
        &mut self,
        orig: &'a str,
        i: usize,
        line: &'a str,
        has_leading_whitespace: bool,
    ) -> bool {
        let is_last = i == count_newlines(orig);

        if let Some(ref mut ib) = self.item_block {
            if ib.add_line(line) {
                return false;
            }
            self.is_prev_line_multi_line = false;
            self.fmt.shape = Shape::legacy(self.max_width, self.fmt_indent);
            let item_fmt = ib.create_string_format(&self.fmt);
            self.result.push_str(&self.comment_line_separator);
            self.result.push_str(&ib.opener);
            match rewrite_string(
                &ib.trimmed_block_as_string(),
                &item_fmt,
                self.max_width.saturating_sub(ib.indent),
            ) {
                Some(s) => self.result.push_str(&Self::join_block(
                    &s,
                    &format!("{}{}", self.comment_line_separator, ib.line_start),
                )),
                None => self.result.push_str(&Self::join_block(
                    &ib.original_block_as_string(),
                    &self.comment_line_separator,
                )),
            };
        } else if self.code_block_attr.is_some() {
            if line.starts_with("```") {
                let code_block = match self.code_block_attr.as_ref().unwrap() {
                    CodeBlockAttribute::Rust
                        if self.fmt.config.format_code_in_doc_comments()
                            && !self.code_block_buffer.is_empty() =>
                    {
                        let mut config = self.fmt.config.clone();
                        config.set().wrap_comments(false);
                        if let Some(s) =
                            crate::format_code_block(&self.code_block_buffer, &config, false)
                        {
                            trim_custom_comment_prefix(&s.snippet)
                        } else {
                            trim_custom_comment_prefix(&self.code_block_buffer)
                        }
                    }
                    _ => trim_custom_comment_prefix(&self.code_block_buffer),
                };
                if !code_block.is_empty() {
                    self.result.push_str(&self.comment_line_separator);
                    self.result
                        .push_str(&Self::join_block(&code_block, &self.comment_line_separator));
                }
                self.code_block_buffer.clear();
                self.result.push_str(&self.comment_line_separator);
                self.result.push_str(line);
                self.code_block_attr = None;
            } else {
                self.code_block_buffer
                    .push_str(&hide_sharp_behind_comment(line));
                self.code_block_buffer.push('\n');
            }
            return false;
        }

        self.code_block_attr = None;
        self.item_block = None;
        if let Some(stripped) = line.strip_prefix("```") {
            self.code_block_attr = Some(CodeBlockAttribute::new(stripped))
        } else if self.fmt.config.wrap_comments() && ItemizedBlock::is_itemized_line(line) {
            let ib = ItemizedBlock::new(line);
            self.item_block = Some(ib);
            return false;
        }

        if self.result == self.opener {
            let force_leading_whitespace = &self.opener == "/* " && count_newlines(orig) == 0;
            if !has_leading_whitespace && !force_leading_whitespace && self.result.ends_with(' ') {
                self.result.pop();
            }
            if line.is_empty() {
                return false;
            }
        } else if self.is_prev_line_multi_line && !line.is_empty() {
            self.result.push(' ')
        } else if is_last && line.is_empty() {
            // trailing blank lines are unwanted
            if !self.closer.is_empty() {
                self.result.push_str(&self.indent_str);
            }
            return true;
        } else {
            self.result.push_str(&self.comment_line_separator);
            if !has_leading_whitespace && self.result.ends_with(' ') {
                self.result.pop();
            }
        }

        if self.fmt.config.wrap_comments()
            && unicode_str_width(line) > self.fmt.shape.width
            && !has_url(line)
        {
            match rewrite_string(line, &self.fmt, self.max_width) {
                Some(ref s) => {
                    self.is_prev_line_multi_line = s.contains('\n');
                    self.result.push_str(s);
                }
                None if self.is_prev_line_multi_line => {
                    // We failed to put the current `line` next to the previous `line`.
                    // Remove the trailing space, then start rewrite on the next line.
                    self.result.pop();
                    self.result.push_str(&self.comment_line_separator);
                    self.fmt.shape = Shape::legacy(self.max_width, self.fmt_indent);
                    match rewrite_string(line, &self.fmt, self.max_width) {
                        Some(ref s) => {
                            self.is_prev_line_multi_line = s.contains('\n');
                            self.result.push_str(s);
                        }
                        None => {
                            self.is_prev_line_multi_line = false;
                            self.result.push_str(line);
                        }
                    }
                }
                None => {
                    self.is_prev_line_multi_line = false;
                    self.result.push_str(line);
                }
            }

            self.fmt.shape = if self.is_prev_line_multi_line {
                // 1 = " "
                let offset = 1 + last_line_width(&self.result) - self.line_start.len();
                Shape {
                    width: self.max_width.saturating_sub(offset),
                    indent: self.fmt_indent,
                    offset: self.fmt.shape.offset + offset,
                }
            } else {
                Shape::legacy(self.max_width, self.fmt_indent)
            };
        } else {
            if line.is_empty() && self.result.ends_with(' ') && !is_last {
                // Remove space if this is an empty comment or a doc comment.
                self.result.pop();
            }
            self.result.push_str(line);
            self.fmt.shape = Shape::legacy(self.max_width, self.fmt_indent);
            self.is_prev_line_multi_line = false;
        }

        false
    }
}

fn rewrite_comment_inner(
    orig: &str,
    block_style: bool,
    style: CommentStyle<'_>,
    shape: Shape,
    config: &Config,
    is_doc_comment: bool,
) -> Option<String> {
    let mut rewriter = CommentRewrite::new(orig, block_style, shape, config);

    let line_breaks = count_newlines(orig.trim_end());
    let lines = orig
        .lines()
        .enumerate()
        .map(|(i, mut line)| {
            line = trim_end_unless_two_whitespaces(line.trim_start(), is_doc_comment);
            // Drop old closer.
            if i == line_breaks && line.ends_with("*/") && !line.starts_with("//") {
                line = line[..(line.len() - 2)].trim_end();
            }

            line
        })
        .map(|s| left_trim_comment_line(s, &style))
        .map(|(line, has_leading_whitespace)| {
            if orig.starts_with("/*") && line_breaks == 0 {
                (
                    line.trim_start(),
                    has_leading_whitespace || config.normalize_comments(),
                )
            } else {
                (line, has_leading_whitespace || config.normalize_comments())
            }
        });

    for (i, (line, has_leading_whitespace)) in lines.enumerate() {
        if rewriter.handle_line(orig, i, line, has_leading_whitespace) {
            break;
        }
    }

    Some(rewriter.finish())
}

const RUSTFMT_CUSTOM_COMMENT_PREFIX: &str = "//#### ";

fn hide_sharp_behind_comment(s: &str) -> Cow<'_, str> {
    let s_trimmed = s.trim();
    if s_trimmed.starts_with("# ") || s_trimmed == "#" {
        Cow::from(format!("{}{}", RUSTFMT_CUSTOM_COMMENT_PREFIX, s))
    } else {
        Cow::from(s)
    }
}

fn trim_custom_comment_prefix(s: &str) -> String {
    s.lines()
        .map(|line| {
            let left_trimmed = line.trim_start();
            if left_trimmed.starts_with(RUSTFMT_CUSTOM_COMMENT_PREFIX) {
                left_trimmed.trim_start_matches(RUSTFMT_CUSTOM_COMMENT_PREFIX)
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Returns `true` if the given string MAY include URLs or alike.
fn has_url(s: &str) -> bool {
    // This function may return false positive, but should get its job done in most cases.
    s.contains("https://") || s.contains("http://") || s.contains("ftp://") || s.contains("file://")
}

/// Given the span, rewrite the missing comment inside it if available.
/// Note that the given span must only include comments (or leading/trailing whitespaces).
pub(crate) fn rewrite_missing_comment(
    span: Span,
    shape: Shape,
    context: &RewriteContext<'_>,
) -> Option<String> {
    let missing_snippet = context.snippet(span);
    let trimmed_snippet = missing_snippet.trim();
    // check the span starts with a comment
    let pos = trimmed_snippet.find('/');
    if !trimmed_snippet.is_empty() && pos.is_some() {
        rewrite_comment(trimmed_snippet, false, shape, context.config)
    } else {
        Some(String::new())
    }
}

/// Recover the missing comments in the specified span, if available.
/// The layout of the comments will be preserved as long as it does not break the code
/// and its total width does not exceed the max width.
pub(crate) fn recover_missing_comment_in_span(
    span: Span,
    shape: Shape,
    context: &RewriteContext<'_>,
    used_width: usize,
) -> Option<String> {
    let missing_comment = rewrite_missing_comment(span, shape, context)?;
    if missing_comment.is_empty() {
        Some(String::new())
    } else {
        let missing_snippet = context.snippet(span);
        let pos = missing_snippet.find('/')?;
        // 1 = ` `
        let total_width = missing_comment.len() + used_width + 1;
        let force_new_line_before_comment =
            missing_snippet[..pos].contains('\n') || total_width > context.config.max_width();
        let sep = if force_new_line_before_comment {
            shape.indent.to_string_with_newline(context.config)
        } else {
            Cow::from(" ")
        };
        Some(format!("{}{}", sep, missing_comment))
    }
}

/// Trim trailing whitespaces unless they consist of two or more whitespaces.
fn trim_end_unless_two_whitespaces(s: &str, is_doc_comment: bool) -> &str {
    if is_doc_comment && s.ends_with("  ") {
        s
    } else {
        s.trim_end()
    }
}

/// Trims whitespace and aligns to indent, but otherwise does not change comments.
fn light_rewrite_comment(
    orig: &str,
    offset: Indent,
    config: &Config,
    is_doc_comment: bool,
) -> String {
    let lines: Vec<&str> = orig
        .lines()
        .map(|l| {
            // This is basically just l.trim(), but in the case that a line starts
            // with `*` we want to leave one space before it, so it aligns with the
            // `*` in `/*`.
            let first_non_whitespace = l.find(|c| !char::is_whitespace(c));
            let left_trimmed = if let Some(fnw) = first_non_whitespace {
                if l.as_bytes()[fnw] == b'*' && fnw > 0 {
                    &l[fnw - 1..]
                } else {
                    &l[fnw..]
                }
            } else {
                ""
            };
            // Preserve markdown's double-space line break syntax in doc comment.
            trim_end_unless_two_whitespaces(left_trimmed, is_doc_comment)
        })
        .collect();
    lines.join(&format!("\n{}", offset.to_string(config)))
}

/// Trims comment characters and possibly a single space from the left of a string.
/// Does not trim all whitespace. If a single space is trimmed from the left of the string,
/// this function returns true.
fn left_trim_comment_line<'a>(line: &'a str, style: &CommentStyle<'_>) -> (&'a str, bool) {
    if line.starts_with("//! ")
        || line.starts_with("/// ")
        || line.starts_with("/*! ")
        || line.starts_with("/** ")
    {
        (&line[4..], true)
    } else if let CommentStyle::Custom(opener) = *style {
        if let Some(stripped) = line.strip_prefix(opener) {
            (stripped, true)
        } else {
            (&line[opener.trim_end().len()..], false)
        }
    } else if line.starts_with("/* ")
        || line.starts_with("// ")
        || line.starts_with("//!")
        || line.starts_with("///")
        || line.starts_with("** ")
        || line.starts_with("/*!")
        || (line.starts_with("/**") && !line.starts_with("/**/"))
    {
        (&line[3..], line.chars().nth(2).unwrap() == ' ')
    } else if line.starts_with("/*")
        || line.starts_with("* ")
        || line.starts_with("//")
        || line.starts_with("**")
    {
        (&line[2..], line.chars().nth(1).unwrap() == ' ')
    } else if let Some(stripped) = line.strip_prefix('*') {
        (stripped, false)
    } else {
        (line, line.starts_with(' '))
    }
}

pub(crate) trait FindUncommented {
    fn find_uncommented(&self, pat: &str) -> Option<usize>;
    fn find_last_uncommented(&self, pat: &str) -> Option<usize>;
}

impl FindUncommented for str {
    fn find_uncommented(&self, pat: &str) -> Option<usize> {
        let mut needle_iter = pat.chars();
        for (kind, (i, b)) in CharClasses::new(self.char_indices()) {
            match needle_iter.next() {
                None => {
                    return Some(i - pat.len());
                }
                Some(c) => match kind {
                    FullCodeCharKind::Normal | FullCodeCharKind::InString if b == c => {}
                    _ => {
                        needle_iter = pat.chars();
                    }
                },
            }
        }

        // Handle case where the pattern is a suffix of the search string
        match needle_iter.next() {
            Some(_) => None,
            None => Some(self.len() - pat.len()),
        }
    }

    fn find_last_uncommented(&self, pat: &str) -> Option<usize> {
        if let Some(left) = self.find_uncommented(pat) {
            let mut result = left;
            // add 1 to use find_last_uncommented for &str after pat
            while let Some(next) = self[(result + 1)..].find_last_uncommented(pat) {
                result += next + 1;
            }
            Some(result)
        } else {
            None
        }
    }
}

// Returns the first byte position after the first comment. The given string
// is expected to be prefixed by a comment, including delimiters.
// Good: `/* /* inner */ outer */ code();`
// Bad:  `code(); // hello\n world!`
pub(crate) fn find_comment_end(s: &str) -> Option<usize> {
    let mut iter = CharClasses::new(s.char_indices());
    for (kind, (i, _c)) in &mut iter {
        if kind == FullCodeCharKind::Normal || kind == FullCodeCharKind::InString {
            return Some(i);
        }
    }

    // Handle case where the comment ends at the end of `s`.
    if iter.status == CharClassesStatus::Normal {
        Some(s.len())
    } else {
        None
    }
}

/// Returns `true` if text contains any comment.
pub(crate) fn contains_comment(text: &str) -> bool {
    CharClasses::new(text.chars()).any(|(kind, _)| kind.is_comment())
}

pub(crate) struct CharClasses<T>
where
    T: Iterator,
    T::Item: RichChar,
{
    base: MultiPeek<T>,
    status: CharClassesStatus,
}

pub(crate) trait RichChar {
    fn get_char(&self) -> char;
}

impl RichChar for char {
    fn get_char(&self) -> char {
        *self
    }
}

impl RichChar for (usize, char) {
    fn get_char(&self) -> char {
        self.1
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum CharClassesStatus {
    Normal,
    /// Character is within a string
    LitString,
    LitStringEscape,
    /// Character is within a raw string
    LitRawString(u32),
    RawStringPrefix(u32),
    RawStringSuffix(u32),
    LitChar,
    LitCharEscape,
    /// Character inside a block comment, with the integer indicating the nesting deepness of the
    /// comment
    BlockComment(u32),
    /// Character inside a block-commented string, with the integer indicating the nesting deepness
    /// of the comment
    StringInBlockComment(u32),
    /// Status when the '/' has been consumed, but not yet the '*', deepness is
    /// the new deepness (after the comment opening).
    BlockCommentOpening(u32),
    /// Status when the '*' has been consumed, but not yet the '/', deepness is
    /// the new deepness (after the comment closing).
    BlockCommentClosing(u32),
    /// Character is within a line comment
    LineComment,
}

/// Distinguish between functional part of code and comments
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub(crate) enum CodeCharKind {
    Normal,
    Comment,
}

/// Distinguish between functional part of code and comments,
/// describing opening and closing of comments for ease when chunking
/// code from tagged characters
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub(crate) enum FullCodeCharKind {
    Normal,
    /// The first character of a comment, there is only one for a comment (always '/')
    StartComment,
    /// Any character inside a comment including the second character of comment
    /// marks ("//", "/*")
    InComment,
    /// Last character of a comment, '\n' for a line comment, '/' for a block comment.
    EndComment,
    /// Start of a mutlitine string inside a comment
    StartStringCommented,
    /// End of a mutlitine string inside a comment
    EndStringCommented,
    /// Inside a commented string
    InStringCommented,
    /// Start of a mutlitine string
    StartString,
    /// End of a mutlitine string
    EndString,
    /// Inside a string.
    InString,
}

impl FullCodeCharKind {
    pub(crate) fn is_comment(self) -> bool {
        match self {
            FullCodeCharKind::StartComment
            | FullCodeCharKind::InComment
            | FullCodeCharKind::EndComment
            | FullCodeCharKind::StartStringCommented
            | FullCodeCharKind::InStringCommented
            | FullCodeCharKind::EndStringCommented => true,
            _ => false,
        }
    }

    /// Returns true if the character is inside a comment
    pub(crate) fn inside_comment(self) -> bool {
        match self {
            FullCodeCharKind::InComment
            | FullCodeCharKind::StartStringCommented
            | FullCodeCharKind::InStringCommented
            | FullCodeCharKind::EndStringCommented => true,
            _ => false,
        }
    }

    pub(crate) fn is_string(self) -> bool {
        self == FullCodeCharKind::InString || self == FullCodeCharKind::StartString
    }

    /// Returns true if the character is within a commented string
    pub(crate) fn is_commented_string(self) -> bool {
        self == FullCodeCharKind::InStringCommented
            || self == FullCodeCharKind::StartStringCommented
    }

    fn to_codecharkind(self) -> CodeCharKind {
        if self.is_comment() {
            CodeCharKind::Comment
        } else {
            CodeCharKind::Normal
        }
    }
}

impl<T> CharClasses<T>
where
    T: Iterator,
    T::Item: RichChar,
{
    pub(crate) fn new(base: T) -> CharClasses<T> {
        CharClasses {
            base: multipeek(base),
            status: CharClassesStatus::Normal,
        }
    }
}

fn is_raw_string_suffix<T>(iter: &mut MultiPeek<T>, count: u32) -> bool
where
    T: Iterator,
    T::Item: RichChar,
{
    for _ in 0..count {
        match iter.peek() {
            Some(c) if c.get_char() == '#' => continue,
            _ => return false,
        }
    }
    true
}

impl<T> Iterator for CharClasses<T>
where
    T: Iterator,
    T::Item: RichChar,
{
    type Item = (FullCodeCharKind, T::Item);

    fn next(&mut self) -> Option<(FullCodeCharKind, T::Item)> {
        let item = self.base.next()?;
        let chr = item.get_char();
        let mut char_kind = FullCodeCharKind::Normal;
        self.status = match self.status {
            CharClassesStatus::LitRawString(sharps) => {
                char_kind = FullCodeCharKind::InString;
                match chr {
                    '"' => {
                        if sharps == 0 {
                            char_kind = FullCodeCharKind::Normal;
                            CharClassesStatus::Normal
                        } else if is_raw_string_suffix(&mut self.base, sharps) {
                            CharClassesStatus::RawStringSuffix(sharps)
                        } else {
                            CharClassesStatus::LitRawString(sharps)
                        }
                    }
                    _ => CharClassesStatus::LitRawString(sharps),
                }
            }
            CharClassesStatus::RawStringPrefix(sharps) => {
                char_kind = FullCodeCharKind::InString;
                match chr {
                    '#' => CharClassesStatus::RawStringPrefix(sharps + 1),
                    '"' => CharClassesStatus::LitRawString(sharps),
                    _ => CharClassesStatus::Normal, // Unreachable.
                }
            }
            CharClassesStatus::RawStringSuffix(sharps) => {
                match chr {
                    '#' => {
                        if sharps == 1 {
                            CharClassesStatus::Normal
                        } else {
                            char_kind = FullCodeCharKind::InString;
                            CharClassesStatus::RawStringSuffix(sharps - 1)
                        }
                    }
                    _ => CharClassesStatus::Normal, // Unreachable
                }
            }
            CharClassesStatus::LitString => {
                char_kind = FullCodeCharKind::InString;
                match chr {
                    '"' => CharClassesStatus::Normal,
                    '\\' => CharClassesStatus::LitStringEscape,
                    _ => CharClassesStatus::LitString,
                }
            }
            CharClassesStatus::LitStringEscape => {
                char_kind = FullCodeCharKind::InString;
                CharClassesStatus::LitString
            }
            CharClassesStatus::LitChar => match chr {
                '\\' => CharClassesStatus::LitCharEscape,
                '\'' => CharClassesStatus::Normal,
                _ => CharClassesStatus::LitChar,
            },
            CharClassesStatus::LitCharEscape => CharClassesStatus::LitChar,
            CharClassesStatus::Normal => match chr {
                'r' => match self.base.peek().map(RichChar::get_char) {
                    Some('#') | Some('"') => {
                        char_kind = FullCodeCharKind::InString;
                        CharClassesStatus::RawStringPrefix(0)
                    }
                    _ => CharClassesStatus::Normal,
                },
                '"' => {
                    char_kind = FullCodeCharKind::InString;
                    CharClassesStatus::LitString
                }
                '\'' => {
                    // HACK: Work around mut borrow.
                    match self.base.peek() {
                        Some(next) if next.get_char() == '\\' => {
                            self.status = CharClassesStatus::LitChar;
                            return Some((char_kind, item));
                        }
                        _ => (),
                    }

                    match self.base.peek() {
                        Some(next) if next.get_char() == '\'' => CharClassesStatus::LitChar,
                        _ => CharClassesStatus::Normal,
                    }
                }
                '/' => match self.base.peek() {
                    Some(next) if next.get_char() == '*' => {
                        self.status = CharClassesStatus::BlockCommentOpening(1);
                        return Some((FullCodeCharKind::StartComment, item));
                    }
                    Some(next) if next.get_char() == '/' => {
                        self.status = CharClassesStatus::LineComment;
                        return Some((FullCodeCharKind::StartComment, item));
                    }
                    _ => CharClassesStatus::Normal,
                },
                _ => CharClassesStatus::Normal,
            },
            CharClassesStatus::StringInBlockComment(deepness) => {
                char_kind = FullCodeCharKind::InStringCommented;
                if chr == '"' {
                    CharClassesStatus::BlockComment(deepness)
                } else if chr == '*' && self.base.peek().map(RichChar::get_char) == Some('/') {
                    char_kind = FullCodeCharKind::InComment;
                    CharClassesStatus::BlockCommentClosing(deepness - 1)
                } else {
                    CharClassesStatus::StringInBlockComment(deepness)
                }
            }
            CharClassesStatus::BlockComment(deepness) => {
                assert_ne!(deepness, 0);
                char_kind = FullCodeCharKind::InComment;
                match self.base.peek() {
                    Some(next) if next.get_char() == '/' && chr == '*' => {
                        CharClassesStatus::BlockCommentClosing(deepness - 1)
                    }
                    Some(next) if next.get_char() == '*' && chr == '/' => {
                        CharClassesStatus::BlockCommentOpening(deepness + 1)
                    }
                    _ if chr == '"' => CharClassesStatus::StringInBlockComment(deepness),
                    _ => self.status,
                }
            }
            CharClassesStatus::BlockCommentOpening(deepness) => {
                assert_eq!(chr, '*');
                self.status = CharClassesStatus::BlockComment(deepness);
                return Some((FullCodeCharKind::InComment, item));
            }
            CharClassesStatus::BlockCommentClosing(deepness) => {
                assert_eq!(chr, '/');
                if deepness == 0 {
                    self.status = CharClassesStatus::Normal;
                    return Some((FullCodeCharKind::EndComment, item));
                } else {
                    self.status = CharClassesStatus::BlockComment(deepness);
                    return Some((FullCodeCharKind::InComment, item));
                }
            }
            CharClassesStatus::LineComment => match chr {
                '\n' => {
                    self.status = CharClassesStatus::Normal;
                    return Some((FullCodeCharKind::EndComment, item));
                }
                _ => {
                    self.status = CharClassesStatus::LineComment;
                    return Some((FullCodeCharKind::InComment, item));
                }
            },
        };
        Some((char_kind, item))
    }
}

/// An iterator over the lines of a string, paired with the char kind at the
/// end of the line.
pub(crate) struct LineClasses<'a> {
    base: iter::Peekable<CharClasses<std::str::Chars<'a>>>,
    kind: FullCodeCharKind,
}

impl<'a> LineClasses<'a> {
    pub(crate) fn new(s: &'a str) -> Self {
        LineClasses {
            base: CharClasses::new(s.chars()).peekable(),
            kind: FullCodeCharKind::Normal,
        }
    }
}

impl<'a> Iterator for LineClasses<'a> {
    type Item = (FullCodeCharKind, String);

    fn next(&mut self) -> Option<Self::Item> {
        self.base.peek()?;

        let mut line = String::new();

        let start_kind = match self.base.peek() {
            Some((kind, _)) => *kind,
            None => unreachable!(),
        };

        for (kind, c) in self.base.by_ref() {
            // needed to set the kind of the ending character on the last line
            self.kind = kind;
            if c == '\n' {
                self.kind = match (start_kind, kind) {
                    (FullCodeCharKind::Normal, FullCodeCharKind::InString) => {
                        FullCodeCharKind::StartString
                    }
                    (FullCodeCharKind::InString, FullCodeCharKind::Normal) => {
                        FullCodeCharKind::EndString
                    }
                    (FullCodeCharKind::InComment, FullCodeCharKind::InStringCommented) => {
                        FullCodeCharKind::StartStringCommented
                    }
                    (FullCodeCharKind::InStringCommented, FullCodeCharKind::InComment) => {
                        FullCodeCharKind::EndStringCommented
                    }
                    _ => kind,
                };
                break;
            }
            line.push(c);
        }

        // Workaround for CRLF newline.
        if line.ends_with('\r') {
            line.pop();
        }

        Some((self.kind, line))
    }
}

/// Iterator over functional and commented parts of a string. Any part of a string is either
/// functional code, either *one* block comment, either *one* line comment. Whitespace between
/// comments is functional code. Line comments contain their ending newlines.
struct UngroupedCommentCodeSlices<'a> {
    slice: &'a str,
    iter: iter::Peekable<CharClasses<std::str::CharIndices<'a>>>,
}

impl<'a> UngroupedCommentCodeSlices<'a> {
    fn new(code: &'a str) -> UngroupedCommentCodeSlices<'a> {
        UngroupedCommentCodeSlices {
            slice: code,
            iter: CharClasses::new(code.char_indices()).peekable(),
        }
    }
}

impl<'a> Iterator for UngroupedCommentCodeSlices<'a> {
    type Item = (CodeCharKind, usize, &'a str);

    fn next(&mut self) -> Option<Self::Item> {
        let (kind, (start_idx, _)) = self.iter.next()?;
        match kind {
            FullCodeCharKind::Normal | FullCodeCharKind::InString => {
                // Consume all the Normal code
                while let Some(&(char_kind, _)) = self.iter.peek() {
                    if char_kind.is_comment() {
                        break;
                    }
                    let _ = self.iter.next();
                }
            }
            FullCodeCharKind::StartComment => {
                // Consume the whole comment
                loop {
                    match self.iter.next() {
                        Some((kind, ..)) if kind.inside_comment() => continue,
                        _ => break,
                    }
                }
            }
            _ => panic!(),
        }
        let slice = match self.iter.peek() {
            Some(&(_, (end_idx, _))) => &self.slice[start_idx..end_idx],
            None => &self.slice[start_idx..],
        };
        Some((
            if kind.is_comment() {
                CodeCharKind::Comment
            } else {
                CodeCharKind::Normal
            },
            start_idx,
            slice,
        ))
    }
}

/// Iterator over an alternating sequence of functional and commented parts of
/// a string. The first item is always a, possibly zero length, subslice of
/// functional text. Line style comments contain their ending newlines.
pub(crate) struct CommentCodeSlices<'a> {
    slice: &'a str,
    last_slice_kind: CodeCharKind,
    last_slice_end: usize,
}

impl<'a> CommentCodeSlices<'a> {
    pub(crate) fn new(slice: &'a str) -> CommentCodeSlices<'a> {
        CommentCodeSlices {
            slice,
            last_slice_kind: CodeCharKind::Comment,
            last_slice_end: 0,
        }
    }
}

impl<'a> Iterator for CommentCodeSlices<'a> {
    type Item = (CodeCharKind, usize, &'a str);

    fn next(&mut self) -> Option<Self::Item> {
        if self.last_slice_end == self.slice.len() {
            return None;
        }

        let mut sub_slice_end = self.last_slice_end;
        let mut first_whitespace = None;
        let subslice = &self.slice[self.last_slice_end..];
        let mut iter = CharClasses::new(subslice.char_indices());

        for (kind, (i, c)) in &mut iter {
            let is_comment_connector = self.last_slice_kind == CodeCharKind::Normal
                && &subslice[..2] == "//"
                && [' ', '\t'].contains(&c);

            if is_comment_connector && first_whitespace.is_none() {
                first_whitespace = Some(i);
            }

            if kind.to_codecharkind() == self.last_slice_kind && !is_comment_connector {
                let last_index = match first_whitespace {
                    Some(j) => j,
                    None => i,
                };
                sub_slice_end = self.last_slice_end + last_index;
                break;
            }

            if !is_comment_connector {
                first_whitespace = None;
            }
        }

        if let (None, true) = (iter.next(), sub_slice_end == self.last_slice_end) {
            // This was the last subslice.
            sub_slice_end = match first_whitespace {
                Some(i) => self.last_slice_end + i,
                None => self.slice.len(),
            };
        }

        let kind = match self.last_slice_kind {
            CodeCharKind::Comment => CodeCharKind::Normal,
            CodeCharKind::Normal => CodeCharKind::Comment,
        };
        let res = (
            kind,
            self.last_slice_end,
            &self.slice[self.last_slice_end..sub_slice_end],
        );
        self.last_slice_end = sub_slice_end;
        self.last_slice_kind = kind;

        Some(res)
    }
}

/// Checks is `new` didn't miss any comment from `span`, if it removed any, return previous text
/// (if it fits in the width/offset, else return `None`), else return `new`
pub(crate) fn recover_comment_removed(
    new: String,
    span: Span,
    context: &RewriteContext<'_>,
) -> Option<String> {
    let snippet = context.snippet(span);
    if snippet != new && changed_comment_content(snippet, &new) {
        // We missed some comments. Warn and keep the original text.
        if context.config.error_on_unformatted() {
            context.report.append(
                context.parse_sess.span_to_filename(span),
                vec![FormattingError::from_span(
                    span,
                    context.parse_sess,
                    ErrorKind::LostComment,
                )],
            );
        }
        Some(snippet.to_owned())
    } else {
        Some(new)
    }
}

pub(crate) fn filter_normal_code(code: &str) -> String {
    let mut buffer = String::with_capacity(code.len());
    LineClasses::new(code).for_each(|(kind, line)| match kind {
        FullCodeCharKind::Normal
        | FullCodeCharKind::StartString
        | FullCodeCharKind::InString
        | FullCodeCharKind::EndString => {
            buffer.push_str(&line);
            buffer.push('\n');
        }
        _ => (),
    });
    if !code.ends_with('\n') && buffer.ends_with('\n') {
        buffer.pop();
    }
    buffer
}

/// Returns `true` if the two strings of code have the same payload of comments.
/// The payload of comments is everything in the string except:
/// - actual code (not comments),
/// - comment start/end marks,
/// - whitespace,
/// - '*' at the beginning of lines in block comments.
fn changed_comment_content(orig: &str, new: &str) -> bool {
    // Cannot write this as a fn since we cannot return types containing closures.
    let code_comment_content = |code| {
        let slices = UngroupedCommentCodeSlices::new(code);
        slices
            .filter(|&(ref kind, _, _)| *kind == CodeCharKind::Comment)
            .flat_map(|(_, _, s)| CommentReducer::new(s))
    };
    let res = code_comment_content(orig).ne(code_comment_content(new));
    debug!(
        "comment::changed_comment_content: {}\norig: '{}'\nnew: '{}'\nraw_old: {}\nraw_new: {}",
        res,
        orig,
        new,
        code_comment_content(orig).collect::<String>(),
        code_comment_content(new).collect::<String>()
    );
    res
}

/// Iterator over the 'payload' characters of a comment.
/// It skips whitespace, comment start/end marks, and '*' at the beginning of lines.
/// The comment must be one comment, ie not more than one start mark (no multiple line comments,
/// for example).
struct CommentReducer<'a> {
    is_block: bool,
    at_start_line: bool,
    iter: std::str::Chars<'a>,
}

impl<'a> CommentReducer<'a> {
    fn new(comment: &'a str) -> CommentReducer<'a> {
        let is_block = comment.starts_with("/*");
        let comment = remove_comment_header(comment);
        CommentReducer {
            is_block,
            // There are no supplementary '*' on the first line.
            at_start_line: false,
            iter: comment.chars(),
        }
    }
}

impl<'a> Iterator for CommentReducer<'a> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut c = self.iter.next()?;
            if self.is_block && self.at_start_line {
                while c.is_whitespace() {
                    c = self.iter.next()?;
                }
                // Ignore leading '*'.
                if c == '*' {
                    c = self.iter.next()?;
                }
            } else if c == '\n' {
                self.at_start_line = true;
            }
            if !c.is_whitespace() {
                return Some(c);
            }
        }
    }
}

fn remove_comment_header(comment: &str) -> &str {
    if comment.starts_with("///") || comment.starts_with("//!") {
        &comment[3..]
    } else if let Some(stripped) = comment.strip_prefix("//") {
        stripped
    } else if (comment.starts_with("/**") && !comment.starts_with("/**/"))
        || comment.starts_with("/*!")
    {
        &comment[3..comment.len() - 2]
    } else {
        assert!(
            comment.starts_with("/*"),
            "string '{}' is not a comment",
            comment
        );
        &comment[2..comment.len() - 2]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::shape::{Indent, Shape};

    #[test]
    fn char_classes() {
        let mut iter = CharClasses::new("//\n\n".chars());

        assert_eq!((FullCodeCharKind::StartComment, '/'), iter.next().unwrap());
        assert_eq!((FullCodeCharKind::InComment, '/'), iter.next().unwrap());
        assert_eq!((FullCodeCharKind::EndComment, '\n'), iter.next().unwrap());
        assert_eq!((FullCodeCharKind::Normal, '\n'), iter.next().unwrap());
        assert_eq!(None, iter.next());
    }

    #[test]
    fn comment_code_slices() {
        let input = "code(); /* test */ 1 + 1";
        let mut iter = CommentCodeSlices::new(input);

        assert_eq!((CodeCharKind::Normal, 0, "code(); "), iter.next().unwrap());
        assert_eq!(
            (CodeCharKind::Comment, 8, "/* test */"),
            iter.next().unwrap()
        );
        assert_eq!((CodeCharKind::Normal, 18, " 1 + 1"), iter.next().unwrap());
        assert_eq!(None, iter.next());
    }

    #[test]
    fn comment_code_slices_two() {
        let input = "// comment\n    test();";
        let mut iter = CommentCodeSlices::new(input);

        assert_eq!((CodeCharKind::Normal, 0, ""), iter.next().unwrap());
        assert_eq!(
            (CodeCharKind::Comment, 0, "// comment\n"),
            iter.next().unwrap()
        );
        assert_eq!(
            (CodeCharKind::Normal, 11, "    test();"),
            iter.next().unwrap()
        );
        assert_eq!(None, iter.next());
    }

    #[test]
    fn comment_code_slices_three() {
        let input = "1 // comment\n    // comment2\n\n";
        let mut iter = CommentCodeSlices::new(input);

        assert_eq!((CodeCharKind::Normal, 0, "1 "), iter.next().unwrap());
        assert_eq!(
            (CodeCharKind::Comment, 2, "// comment\n    // comment2\n"),
            iter.next().unwrap()
        );
        assert_eq!((CodeCharKind::Normal, 29, "\n"), iter.next().unwrap());
        assert_eq!(None, iter.next());
    }

    #[test]
    #[rustfmt::skip]
    fn format_doc_comments() {
        let mut wrap_normalize_config: crate::config::Config = Default::default();
        wrap_normalize_config.set().wrap_comments(true);
        wrap_normalize_config.set().normalize_comments(true);

        let mut wrap_config: crate::config::Config = Default::default();
        wrap_config.set().wrap_comments(true);

        let comment = rewrite_comment(" //test",
                                      true,
                                      Shape::legacy(100, Indent::new(0, 100)),
                                      &wrap_normalize_config).unwrap();
        assert_eq!("/* test */", comment);

        let comment = rewrite_comment("// comment on a",
                                      false,
                                      Shape::legacy(10, Indent::empty()),
                                      &wrap_normalize_config).unwrap();
        assert_eq!("// comment\n// on a", comment);

        let comment = rewrite_comment("//  A multi line comment\n             // between args.",
                                      false,
                                      Shape::legacy(60, Indent::new(0, 12)),
                                      &wrap_normalize_config).unwrap();
        assert_eq!("//  A multi line comment\n            // between args.", comment);

        let input = "// comment";
        let expected =
            "/* comment */";
        let comment = rewrite_comment(input,
                                      true,
                                      Shape::legacy(9, Indent::new(0, 69)),
                                      &wrap_normalize_config).unwrap();
        assert_eq!(expected, comment);

        let comment = rewrite_comment("/*   trimmed    */",
                                      true,
                                      Shape::legacy(100, Indent::new(0, 100)),
                                      &wrap_normalize_config).unwrap();
        assert_eq!("/* trimmed */", comment);

        // Check that different comment style are properly recognised.
        let comment = rewrite_comment(r#"/// test1
                                         /// test2
                                         /*
                                          * test3
                                          */"#,
                                      false,
                                      Shape::legacy(100, Indent::new(0, 0)),
                                      &wrap_normalize_config).unwrap();
        assert_eq!("/// test1\n/// test2\n// test3", comment);

        // Check that the blank line marks the end of a commented paragraph.
        let comment = rewrite_comment(r#"// test1

                                         // test2"#,
                                      false,
                                      Shape::legacy(100, Indent::new(0, 0)),
                                      &wrap_normalize_config).unwrap();
        assert_eq!("// test1\n\n// test2", comment);

        // Check that the blank line marks the end of a custom-commented paragraph.
        let comment = rewrite_comment(r#"//@ test1

                                         //@ test2"#,
                                      false,
                                      Shape::legacy(100, Indent::new(0, 0)),
                                      &wrap_normalize_config).unwrap();
        assert_eq!("//@ test1\n\n//@ test2", comment);

        // Check that bare lines are just indented but otherwise left unchanged.
        let comment = rewrite_comment(r#"// test1
                                         /*
                                           a bare line!

                                                another bare line!
                                          */"#,
                                      false,
                                      Shape::legacy(100, Indent::new(0, 0)),
                                      &wrap_config).unwrap();
        assert_eq!("// test1\n/*\n a bare line!\n\n      another bare line!\n*/", comment);
    }

    // This is probably intended to be a non-test fn, but it is not used.
    // We should keep this around unless it helps us test stuff to remove it.
    fn uncommented(text: &str) -> String {
        CharClasses::new(text.chars())
            .filter_map(|(s, c)| match s {
                FullCodeCharKind::Normal | FullCodeCharKind::InString => Some(c),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn test_uncommented() {
        assert_eq!(&uncommented("abc/*...*/"), "abc");
        assert_eq!(
            &uncommented("// .... /* \n../* /* *** / */ */a/* // */c\n"),
            "..ac\n"
        );
        assert_eq!(&uncommented("abc \" /* */\" qsdf"), "abc \" /* */\" qsdf");
    }

    #[test]
    fn test_contains_comment() {
        assert_eq!(contains_comment("abc"), false);
        assert_eq!(contains_comment("abc // qsdf"), true);
        assert_eq!(contains_comment("abc /* kqsdf"), true);
        assert_eq!(contains_comment("abc \" /* */\" qsdf"), false);
    }

    #[test]
    fn test_find_uncommented() {
        fn check(haystack: &str, needle: &str, expected: Option<usize>) {
            assert_eq!(expected, haystack.find_uncommented(needle));
        }

        check("/*/ */test", "test", Some(6));
        check("//test\ntest", "test", Some(7));
        check("/* comment only */", "whatever", None);
        check(
            "/* comment */ some text /* more commentary */ result",
            "result",
            Some(46),
        );
        check("sup // sup", "p", Some(2));
        check("sup", "x", None);
        check(r#"π? /**/ π is nice!"#, r#"π is nice"#, Some(9));
        check("/*sup yo? \n sup*/ sup", "p", Some(20));
        check("hel/*lohello*/lo", "hello", None);
        check("acb", "ab", None);
        check(",/*A*/ ", ",", Some(0));
        check("abc", "abc", Some(0));
        check("/* abc */", "abc", None);
        check("/**/abc/* */", "abc", Some(4));
        check("\"/* abc */\"", "abc", Some(4));
        check("\"/* abc", "abc", Some(4));
    }

    #[test]
    fn test_filter_normal_code() {
        let s = r#"
fn main() {
    println!("hello, world");
}
"#;
        assert_eq!(s, filter_normal_code(s));
        let s_with_comment = r#"
fn main() {
    // hello, world
    println!("hello, world");
}
"#;
        assert_eq!(s, filter_normal_code(s_with_comment));
    }
}
