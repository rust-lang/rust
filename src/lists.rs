// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp;
use std::iter::Peekable;

use syntax::codemap::{CodeMap, BytePos};

use {Indent, Shape};
use comment::{FindUncommented, rewrite_comment, find_comment_end};
use config::{Config, IndentStyle};
use rewrite::RewriteContext;
use utils::mk_sp;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
/// Formatting tactic for lists. This will be cast down to a
/// DefinitiveListTactic depending on the number and length of the items and
/// their comments.
pub enum ListTactic {
    // One item per row.
    Vertical,
    // All items on one row.
    Horizontal,
    // Try Horizontal layout, if that fails then vertical.
    HorizontalVertical,
    // HorizontalVertical with a soft limit of n characters.
    LimitedHorizontalVertical(usize),
    // Pack as many items as possible per row over (possibly) many rows.
    Mixed,
}

impl_enum_serialize_and_deserialize!(ListTactic, Vertical, Horizontal, HorizontalVertical, Mixed);

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum SeparatorTactic {
    Always,
    Never,
    Vertical,
}

impl_enum_serialize_and_deserialize!(SeparatorTactic, Always, Never, Vertical);

impl SeparatorTactic {
    pub fn from_bool(b: bool) -> SeparatorTactic {
        if b {
            SeparatorTactic::Always
        } else {
            SeparatorTactic::Never
        }
    }
}

pub struct ListFormatting<'a> {
    pub tactic: DefinitiveListTactic,
    pub separator: &'a str,
    pub trailing_separator: SeparatorTactic,
    pub shape: Shape,
    // Non-expressions, e.g. items, will have a new line at the end of the list.
    // Important for comment styles.
    pub ends_with_newline: bool,
    pub config: &'a Config,
}

pub fn format_fn_args<I>(items: I, shape: Shape, config: &Config) -> Option<String>
where
    I: Iterator<Item = ListItem>,
{
    list_helper(
        items,
        shape,
        config,
        ListTactic::LimitedHorizontalVertical(config.fn_call_width()),
    )
}

pub fn format_item_list<I>(items: I, shape: Shape, config: &Config) -> Option<String>
where
    I: Iterator<Item = ListItem>,
{
    list_helper(items, shape, config, ListTactic::HorizontalVertical)
}

pub fn list_helper<I>(items: I, shape: Shape, config: &Config, tactic: ListTactic) -> Option<String>
where
    I: Iterator<Item = ListItem>,
{
    let item_vec: Vec<_> = items.collect();
    let tactic = definitive_tactic(&item_vec, tactic, shape.width);
    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: SeparatorTactic::Never,
        shape: shape,
        ends_with_newline: false,
        config: config,
    };

    write_list(&item_vec, &fmt)
}

impl AsRef<ListItem> for ListItem {
    fn as_ref(&self) -> &ListItem {
        self
    }
}

pub struct ListItem {
    // None for comments mean that they are not present.
    pub pre_comment: Option<String>,
    // Item should include attributes and doc comments. None indicates a failed
    // rewrite.
    pub item: Option<String>,
    pub post_comment: Option<String>,
    // Whether there is extra whitespace before this item.
    pub new_lines: bool,
}

impl ListItem {
    pub fn is_multiline(&self) -> bool {
        self.item.as_ref().map_or(false, |s| s.contains('\n')) || self.pre_comment.is_some() ||
            self.post_comment.as_ref().map_or(
                false,
                |s| s.contains('\n'),
            )
    }

    pub fn has_line_pre_comment(&self) -> bool {
        self.pre_comment.as_ref().map_or(false, |comment| {
            comment.starts_with("//")
        })
    }

    pub fn from_str<S: Into<String>>(s: S) -> ListItem {
        ListItem {
            pre_comment: None,
            item: Some(s.into()),
            post_comment: None,
            new_lines: false,
        }
    }
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
/// The definitive formatting tactic for lists.
pub enum DefinitiveListTactic {
    Vertical,
    Horizontal,
    Mixed,
}

pub fn definitive_tactic<I, T>(items: I, tactic: ListTactic, width: usize) -> DefinitiveListTactic
where
    I: IntoIterator<Item = T> + Clone,
    T: AsRef<ListItem>,
{
    let pre_line_comments = items.clone().into_iter().any(|item| {
        item.as_ref().has_line_pre_comment()
    });

    let limit = match tactic {
        _ if pre_line_comments => return DefinitiveListTactic::Vertical,
        ListTactic::Mixed => return DefinitiveListTactic::Mixed,
        ListTactic::Horizontal => return DefinitiveListTactic::Horizontal,
        ListTactic::Vertical => return DefinitiveListTactic::Vertical,
        ListTactic::LimitedHorizontalVertical(limit) => ::std::cmp::min(width, limit),
        ListTactic::HorizontalVertical => width,
    };

    let (sep_count, total_width) = calculate_width(items.clone());
    let sep_len = ", ".len(); // FIXME: make more generic?
    let total_sep_len = sep_len * sep_count.checked_sub(1).unwrap_or(0);
    let real_total = total_width + total_sep_len;

    if real_total <= limit && !pre_line_comments &&
        !items.into_iter().any(|item| item.as_ref().is_multiline())
    {
        DefinitiveListTactic::Horizontal
    } else {
        DefinitiveListTactic::Vertical
    }
}

// Format a list of commented items into a string.
// TODO: add unit tests
pub fn write_list<I, T>(items: I, formatting: &ListFormatting) -> Option<String>
where
    I: IntoIterator<Item = T>,
    T: AsRef<ListItem>,
{
    let tactic = formatting.tactic;
    let sep_len = formatting.separator.len();

    // Now that we know how we will layout, we can decide for sure if there
    // will be a trailing separator.
    let trailing_separator = needs_trailing_separator(formatting.trailing_separator, tactic);
    let mut result = String::new();
    let mut iter = items.into_iter().enumerate().peekable();

    let mut line_len = 0;
    let indent_str = &formatting.shape.indent.to_string(formatting.config);
    while let Some((i, item)) = iter.next() {
        let item = item.as_ref();
        let inner_item = try_opt!(item.item.as_ref());
        let first = i == 0;
        let last = iter.peek().is_none();
        let separate = !last || trailing_separator;
        let item_sep_len = if separate { sep_len } else { 0 };

        // Item string may be multi-line. Its length (used for block comment alignment)
        // should be only the length of the last line.
        let item_last_line = if item.is_multiline() {
            inner_item.lines().last().unwrap_or("")
        } else {
            inner_item.as_ref()
        };
        let mut item_last_line_width = item_last_line.len() + item_sep_len;
        if item_last_line.starts_with(indent_str) {
            item_last_line_width -= indent_str.len();
        }

        match tactic {
            DefinitiveListTactic::Horizontal if !first => {
                result.push(' ');
            }
            DefinitiveListTactic::Vertical if !first => {
                result.push('\n');
                result.push_str(indent_str);
            }
            DefinitiveListTactic::Mixed => {
                let total_width = total_item_width(item) + item_sep_len;

                // 1 is space between separator and item.
                if line_len > 0 && line_len + 1 + total_width > formatting.shape.width {
                    result.push('\n');
                    result.push_str(indent_str);
                    line_len = 0;
                }

                if line_len > 0 {
                    result.push(' ');
                    line_len += 1;
                }

                line_len += total_width;
            }
            _ => {}
        }

        // Pre-comments
        if let Some(ref comment) = item.pre_comment {
            // Block style in non-vertical mode.
            let block_mode = tactic != DefinitiveListTactic::Vertical;
            // Width restriction is only relevant in vertical mode.
            let comment = try_opt!(rewrite_comment(
                comment,
                block_mode,
                formatting.shape,
                formatting.config,
            ));
            result.push_str(&comment);

            if tactic == DefinitiveListTactic::Vertical {
                result.push('\n');
                result.push_str(indent_str);
            } else {
                result.push(' ');
            }
        }

        result.push_str(&inner_item[..]);

        // Post-comments
        if tactic != DefinitiveListTactic::Vertical && item.post_comment.is_some() {
            let comment = item.post_comment.as_ref().unwrap();
            let formatted_comment = try_opt!(rewrite_comment(
                comment,
                true,
                Shape::legacy(formatting.shape.width, Indent::empty()),
                formatting.config,
            ));

            result.push(' ');
            result.push_str(&formatted_comment);
        }

        if separate {
            result.push_str(formatting.separator);
        }

        if tactic == DefinitiveListTactic::Vertical && item.post_comment.is_some() {
            // 1 = space between item and comment.
            let width = formatting
                .shape
                .width
                .checked_sub(item_last_line_width + 1)
                .unwrap_or(1);
            let mut offset = formatting.shape.indent;
            offset.alignment += item_last_line_width + 1;
            let comment = item.post_comment.as_ref().unwrap();

            debug!("Width = {}, offset = {:?}", width, offset);
            // Use block-style only for the last item or multiline comments.
            let block_style = !formatting.ends_with_newline && last ||
                comment.trim().contains('\n') ||
                comment.trim().len() > width;

            let formatted_comment = try_opt!(rewrite_comment(
                comment,
                block_style,
                Shape::legacy(width, offset),
                formatting.config,
            ));

            if !formatted_comment.starts_with('\n') {
                result.push(' ');
            }
            result.push_str(&formatted_comment);
        }

        if !last && tactic == DefinitiveListTactic::Vertical && item.new_lines {
            result.push('\n');
        }
    }

    Some(result)
}

pub struct ListItems<'a, I, F1, F2, F3>
where
    I: Iterator,
{
    codemap: &'a CodeMap,
    inner: Peekable<I>,
    get_lo: F1,
    get_hi: F2,
    get_item_string: F3,
    prev_span_end: BytePos,
    next_span_start: BytePos,
    terminator: &'a str,
}

impl<'a, T, I, F1, F2, F3> Iterator for ListItems<'a, I, F1, F2, F3>
where
    I: Iterator<Item = T>,
    F1: Fn(&T) -> BytePos,
    F2: Fn(&T) -> BytePos,
    F3: Fn(&T) -> Option<String>,
{
    type Item = ListItem;

    fn next(&mut self) -> Option<Self::Item> {
        let white_space: &[_] = &[' ', '\t'];

        self.inner.next().map(|item| {
            let mut new_lines = false;
            // Pre-comment
            let pre_snippet = self.codemap
                .span_to_snippet(mk_sp(self.prev_span_end, (self.get_lo)(&item)))
                .unwrap();
            let trimmed_pre_snippet = pre_snippet.trim();
            let has_pre_comment = trimmed_pre_snippet.contains("//") ||
                trimmed_pre_snippet.contains("/*");
            let pre_comment = if has_pre_comment {
                Some(trimmed_pre_snippet.to_owned())
            } else {
                None
            };

            // Post-comment
            let next_start = match self.inner.peek() {
                Some(next_item) => (self.get_lo)(next_item),
                None => self.next_span_start,
            };
            let post_snippet = self.codemap
                .span_to_snippet(mk_sp((self.get_hi)(&item), next_start))
                .unwrap();

            let comment_end = match self.inner.peek() {
                Some(..) => {
                    let mut block_open_index = post_snippet.find("/*");
                    // check if it realy is a block comment (and not //*)
                    if let Some(i) = block_open_index {
                        if i > 0 && &post_snippet[i - 1..i] == "/" {
                            block_open_index = None;
                        }
                    }
                    let newline_index = post_snippet.find('\n');
                    let separator_index = post_snippet.find_uncommented(",").unwrap();

                    match (block_open_index, newline_index) {
                        // Separator before comment, with the next item on same line.
                        // Comment belongs to next item.
                        (Some(i), None) if i > separator_index => separator_index + 1,
                        // Block-style post-comment before the separator.
                        (Some(i), None) => {
                            cmp::max(
                                find_comment_end(&post_snippet[i..]).unwrap() + i,
                                separator_index + 1,
                            )
                        }
                        // Block-style post-comment. Either before or after the separator.
                        (Some(i), Some(j)) if i < j => {
                            cmp::max(
                                find_comment_end(&post_snippet[i..]).unwrap() + i,
                                separator_index + 1,
                            )
                        }
                        // Potential *single* line comment.
                        (_, Some(j)) if j > separator_index => j + 1,
                        _ => post_snippet.len(),
                    }
                }
                None => {
                    post_snippet.find_uncommented(self.terminator).unwrap_or(
                        post_snippet
                            .len(),
                    )
                }
            };

            if !post_snippet.is_empty() && comment_end > 0 {
                // Account for extra whitespace between items. This is fiddly
                // because of the way we divide pre- and post- comments.

                // Everything from the separator to the next item.
                let test_snippet = &post_snippet[comment_end - 1..];
                let first_newline = test_snippet.find('\n').unwrap_or(test_snippet.len());
                // From the end of the first line of comments.
                let test_snippet = &test_snippet[first_newline..];
                let first = test_snippet.find(|c: char| !c.is_whitespace()).unwrap_or(
                    test_snippet
                        .len(),
                );
                // From the end of the first line of comments to the next non-whitespace char.
                let test_snippet = &test_snippet[..first];

                if test_snippet.chars().filter(|c| c == &'\n').count() > 1 {
                    // There were multiple line breaks which got trimmed to nothing.
                    new_lines = true;
                }
            }

            // Cleanup post-comment: strip separators and whitespace.
            self.prev_span_end = (self.get_hi)(&item) + BytePos(comment_end as u32);
            let post_snippet = post_snippet[..comment_end].trim();

            let post_snippet_trimmed = if post_snippet.starts_with(',') {
                post_snippet[1..].trim_matches(white_space)
            } else if post_snippet.ends_with(',') {
                post_snippet[..(post_snippet.len() - 1)].trim_matches(white_space)
            } else {
                post_snippet
            };

            let post_comment = if !post_snippet_trimmed.is_empty() {
                Some(post_snippet_trimmed.to_owned())
            } else {
                None
            };

            ListItem {
                pre_comment: pre_comment,
                item: (self.get_item_string)(&item),
                post_comment: post_comment,
                new_lines: new_lines,
            }
        })
    }
}

// Creates an iterator over a list's items with associated comments.
pub fn itemize_list<'a, T, I, F1, F2, F3>(
    codemap: &'a CodeMap,
    inner: I,
    terminator: &'a str,
    get_lo: F1,
    get_hi: F2,
    get_item_string: F3,
    prev_span_end: BytePos,
    next_span_start: BytePos,
) -> ListItems<'a, I, F1, F2, F3>
where
    I: Iterator<Item = T>,
    F1: Fn(&T) -> BytePos,
    F2: Fn(&T) -> BytePos,
    F3: Fn(&T) -> Option<String>,
{
    ListItems {
        codemap: codemap,
        inner: inner.peekable(),
        get_lo: get_lo,
        get_hi: get_hi,
        get_item_string: get_item_string,
        prev_span_end: prev_span_end,
        next_span_start: next_span_start,
        terminator: terminator,
    }
}

fn needs_trailing_separator(
    separator_tactic: SeparatorTactic,
    list_tactic: DefinitiveListTactic,
) -> bool {
    match separator_tactic {
        SeparatorTactic::Always => true,
        SeparatorTactic::Vertical => list_tactic == DefinitiveListTactic::Vertical,
        SeparatorTactic::Never => false,
    }
}

/// Returns the count and total width of the list items.
fn calculate_width<I, T>(items: I) -> (usize, usize)
where
    I: IntoIterator<Item = T>,
    T: AsRef<ListItem>,
{
    items
        .into_iter()
        .map(|item| total_item_width(item.as_ref()))
        .fold((0, 0), |acc, l| (acc.0 + 1, acc.1 + l))
}

fn total_item_width(item: &ListItem) -> usize {
    comment_len(item.pre_comment.as_ref().map(|x| &(*x)[..])) +
        comment_len(item.post_comment.as_ref().map(|x| &(*x)[..])) +
        item.item.as_ref().map_or(0, |str| str.len())
}

fn comment_len(comment: Option<&str>) -> usize {
    match comment {
        Some(s) => {
            let text_len = s.trim().len();
            if text_len > 0 {
                // We'll put " /*" before and " */" after inline comments.
                text_len + 6
            } else {
                text_len
            }
        }
        None => 0,
    }
}

// Compute horizontal and vertical shapes for a struct-lit-like thing.
pub fn struct_lit_shape(
    shape: Shape,
    context: &RewriteContext,
    prefix_width: usize,
    suffix_width: usize,
) -> Option<(Option<Shape>, Shape)> {
    let v_shape = match context.config.struct_lit_style() {
        IndentStyle::Visual => {
            try_opt!(
                try_opt!(shape.visual_indent(0).shrink_left(prefix_width)).sub_width(suffix_width)
            )
        }
        IndentStyle::Block => {
            let shape = shape.block_indent(context.config.tab_spaces());
            Shape {
                width: try_opt!(context.config.max_width().checked_sub(shape.indent.width())),
                ..shape
            }
        }
    };
    let h_shape = shape.sub_width(prefix_width + suffix_width);
    Some((h_shape, v_shape))
}

// Compute the tactic for the internals of a struct-lit-like thing.
pub fn struct_lit_tactic(
    h_shape: Option<Shape>,
    context: &RewriteContext,
    items: &[ListItem],
) -> DefinitiveListTactic {
    if let Some(h_shape) = h_shape {
        let mut prelim_tactic = match (context.config.struct_lit_style(), items.len()) {
            (IndentStyle::Visual, 1) => ListTactic::HorizontalVertical,
            _ => context.config.struct_lit_multiline_style().to_list_tactic(),
        };

        if prelim_tactic == ListTactic::HorizontalVertical && items.len() > 1 {
            prelim_tactic =
                ListTactic::LimitedHorizontalVertical(context.config.struct_lit_width());
        }

        definitive_tactic(items, prelim_tactic, h_shape.width)
    } else {
        DefinitiveListTactic::Vertical
    }
}

// Given a tactic and possible shapes for horizontal and vertical layout,
// come up with the actual shape to use.
pub fn shape_for_tactic(
    tactic: DefinitiveListTactic,
    h_shape: Option<Shape>,
    v_shape: Shape,
) -> Shape {
    match tactic {
        DefinitiveListTactic::Horizontal => h_shape.unwrap(),
        _ => v_shape,
    }
}

// Create a ListFormatting object for formatting the internals of a
// struct-lit-like thing, that is a series of fields.
pub fn struct_lit_formatting<'a>(
    shape: Shape,
    tactic: DefinitiveListTactic,
    context: &'a RewriteContext,
    force_no_trailing_comma: bool,
) -> ListFormatting<'a> {
    let ends_with_newline = context.config.struct_lit_style() != IndentStyle::Visual &&
        tactic == DefinitiveListTactic::Vertical;
    ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: if force_no_trailing_comma {
            SeparatorTactic::Never
        } else {
            context.config.trailing_comma()
        },
        shape: shape,
        ends_with_newline: ends_with_newline,
        config: context.config,
    }
}
