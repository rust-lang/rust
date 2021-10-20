//! Format list-like expressions and items.

use std::cmp;
use std::iter::Peekable;

use rustc_span::BytePos;

use crate::comment::{find_comment_end, rewrite_comment, FindUncommented};
use crate::config::lists::*;
use crate::config::{Config, IndentStyle};
use crate::rewrite::RewriteContext;
use crate::shape::{Indent, Shape};
use crate::utils::{
    count_newlines, first_line_width, last_line_width, mk_sp, starts_with_newline,
    unicode_str_width,
};
use crate::visitor::SnippetProvider;

pub(crate) struct ListFormatting<'a> {
    tactic: DefinitiveListTactic,
    separator: &'a str,
    trailing_separator: SeparatorTactic,
    separator_place: SeparatorPlace,
    shape: Shape,
    // Non-expressions, e.g., items, will have a new line at the end of the list.
    // Important for comment styles.
    ends_with_newline: bool,
    // Remove newlines between list elements for expressions.
    preserve_newline: bool,
    // Nested import lists get some special handling for the "Mixed" list type
    nested: bool,
    // Whether comments should be visually aligned.
    align_comments: bool,
    config: &'a Config,
}

impl<'a> ListFormatting<'a> {
    pub(crate) fn new(shape: Shape, config: &'a Config) -> Self {
        ListFormatting {
            tactic: DefinitiveListTactic::Vertical,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            separator_place: SeparatorPlace::Back,
            shape,
            ends_with_newline: true,
            preserve_newline: false,
            nested: false,
            align_comments: true,
            config,
        }
    }

    pub(crate) fn tactic(mut self, tactic: DefinitiveListTactic) -> Self {
        self.tactic = tactic;
        self
    }

    pub(crate) fn separator(mut self, separator: &'a str) -> Self {
        self.separator = separator;
        self
    }

    pub(crate) fn trailing_separator(mut self, trailing_separator: SeparatorTactic) -> Self {
        self.trailing_separator = trailing_separator;
        self
    }

    pub(crate) fn separator_place(mut self, separator_place: SeparatorPlace) -> Self {
        self.separator_place = separator_place;
        self
    }

    pub(crate) fn ends_with_newline(mut self, ends_with_newline: bool) -> Self {
        self.ends_with_newline = ends_with_newline;
        self
    }

    pub(crate) fn preserve_newline(mut self, preserve_newline: bool) -> Self {
        self.preserve_newline = preserve_newline;
        self
    }

    pub(crate) fn nested(mut self, nested: bool) -> Self {
        self.nested = nested;
        self
    }

    pub(crate) fn align_comments(mut self, align_comments: bool) -> Self {
        self.align_comments = align_comments;
        self
    }

    pub(crate) fn needs_trailing_separator(&self) -> bool {
        match self.trailing_separator {
            // We always put separator in front.
            SeparatorTactic::Always => true,
            SeparatorTactic::Vertical => self.tactic == DefinitiveListTactic::Vertical,
            SeparatorTactic::Never => {
                self.tactic == DefinitiveListTactic::Vertical && self.separator_place.is_front()
            }
        }
    }
}

impl AsRef<ListItem> for ListItem {
    fn as_ref(&self) -> &ListItem {
        self
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub(crate) enum ListItemCommentStyle {
    // Try to keep the comment on the same line with the item.
    SameLine,
    // Put the comment on the previous or the next line of the item.
    DifferentLine,
    // No comment available.
    None,
}

#[derive(Debug, Clone)]
pub(crate) struct ListItem {
    // None for comments mean that they are not present.
    pub(crate) pre_comment: Option<String>,
    pub(crate) pre_comment_style: ListItemCommentStyle,
    // Item should include attributes and doc comments. None indicates a failed
    // rewrite.
    pub(crate) item: Option<String>,
    pub(crate) post_comment: Option<String>,
    // Whether there is extra whitespace before this item.
    pub(crate) new_lines: bool,
}

impl ListItem {
    pub(crate) fn empty() -> ListItem {
        ListItem {
            pre_comment: None,
            pre_comment_style: ListItemCommentStyle::None,
            item: None,
            post_comment: None,
            new_lines: false,
        }
    }

    pub(crate) fn inner_as_ref(&self) -> &str {
        self.item.as_ref().map_or("", |s| s)
    }

    pub(crate) fn is_different_group(&self) -> bool {
        self.inner_as_ref().contains('\n')
            || self.pre_comment.is_some()
            || self
                .post_comment
                .as_ref()
                .map_or(false, |s| s.contains('\n'))
    }

    pub(crate) fn is_multiline(&self) -> bool {
        self.inner_as_ref().contains('\n')
            || self
                .pre_comment
                .as_ref()
                .map_or(false, |s| s.contains('\n'))
            || self
                .post_comment
                .as_ref()
                .map_or(false, |s| s.contains('\n'))
    }

    pub(crate) fn has_single_line_comment(&self) -> bool {
        self.pre_comment
            .as_ref()
            .map_or(false, |comment| comment.trim_start().starts_with("//"))
            || self
                .post_comment
                .as_ref()
                .map_or(false, |comment| comment.trim_start().starts_with("//"))
    }

    pub(crate) fn has_comment(&self) -> bool {
        self.pre_comment.is_some() || self.post_comment.is_some()
    }

    pub(crate) fn from_str<S: Into<String>>(s: S) -> ListItem {
        ListItem {
            pre_comment: None,
            pre_comment_style: ListItemCommentStyle::None,
            item: Some(s.into()),
            post_comment: None,
            new_lines: false,
        }
    }

    // Returns `true` if the item causes something to be written.
    fn is_substantial(&self) -> bool {
        fn empty(s: &Option<String>) -> bool {
            !matches!(*s, Some(ref s) if !s.is_empty())
        }

        !(empty(&self.pre_comment) && empty(&self.item) && empty(&self.post_comment))
    }
}

/// The type of separator for lists.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum Separator {
    Comma,
    VerticalBar,
}

impl Separator {
    pub(crate) fn len(self) -> usize {
        match self {
            // 2 = `, `
            Separator::Comma => 2,
            // 3 = ` | `
            Separator::VerticalBar => 3,
        }
    }
}

pub(crate) fn definitive_tactic<I, T>(
    items: I,
    tactic: ListTactic,
    sep: Separator,
    width: usize,
) -> DefinitiveListTactic
where
    I: IntoIterator<Item = T> + Clone,
    T: AsRef<ListItem>,
{
    let pre_line_comments = items
        .clone()
        .into_iter()
        .any(|item| item.as_ref().has_single_line_comment());

    let limit = match tactic {
        _ if pre_line_comments => return DefinitiveListTactic::Vertical,
        ListTactic::Horizontal => return DefinitiveListTactic::Horizontal,
        ListTactic::Vertical => return DefinitiveListTactic::Vertical,
        ListTactic::LimitedHorizontalVertical(limit) => ::std::cmp::min(width, limit),
        ListTactic::Mixed | ListTactic::HorizontalVertical => width,
    };

    let (sep_count, total_width) = calculate_width(items.clone());
    let total_sep_len = sep.len() * sep_count.saturating_sub(1);
    let real_total = total_width + total_sep_len;

    if real_total <= limit && !items.into_iter().any(|item| item.as_ref().is_multiline()) {
        DefinitiveListTactic::Horizontal
    } else {
        match tactic {
            ListTactic::Mixed => DefinitiveListTactic::Mixed,
            _ => DefinitiveListTactic::Vertical,
        }
    }
}

// Format a list of commented items into a string.
pub(crate) fn write_list<I, T>(items: I, formatting: &ListFormatting<'_>) -> Option<String>
where
    I: IntoIterator<Item = T> + Clone,
    T: AsRef<ListItem>,
{
    let tactic = formatting.tactic;
    let sep_len = formatting.separator.len();

    // Now that we know how we will layout, we can decide for sure if there
    // will be a trailing separator.
    let mut trailing_separator = formatting.needs_trailing_separator();
    let mut result = String::with_capacity(128);
    let cloned_items = items.clone();
    let mut iter = items.into_iter().enumerate().peekable();
    let mut item_max_width: Option<usize> = None;
    let sep_place =
        SeparatorPlace::from_tactic(formatting.separator_place, tactic, formatting.separator);
    let mut prev_item_had_post_comment = false;
    let mut prev_item_is_nested_import = false;

    let mut line_len = 0;
    let indent_str = &formatting.shape.indent.to_string(formatting.config);
    while let Some((i, item)) = iter.next() {
        let item = item.as_ref();
        let inner_item = item.item.as_ref()?;
        let first = i == 0;
        let last = iter.peek().is_none();
        let mut separate = match sep_place {
            SeparatorPlace::Front => !first,
            SeparatorPlace::Back => !last || trailing_separator,
        };
        let item_sep_len = if separate { sep_len } else { 0 };

        // Item string may be multi-line. Its length (used for block comment alignment)
        // should be only the length of the last line.
        let item_last_line = if item.is_multiline() {
            inner_item.lines().last().unwrap_or("")
        } else {
            inner_item.as_ref()
        };
        let mut item_last_line_width = item_last_line.len() + item_sep_len;
        if item_last_line.starts_with(&**indent_str) {
            item_last_line_width -= indent_str.len();
        }

        if !item.is_substantial() {
            continue;
        }

        match tactic {
            DefinitiveListTactic::Horizontal if !first => {
                result.push(' ');
            }
            DefinitiveListTactic::SpecialMacro(num_args_before) => {
                if i == 0 {
                    // Nothing
                } else if i < num_args_before {
                    result.push(' ');
                } else if i <= num_args_before + 1 {
                    result.push('\n');
                    result.push_str(indent_str);
                } else {
                    result.push(' ');
                }
            }
            DefinitiveListTactic::Vertical
                if !first && !inner_item.is_empty() && !result.is_empty() =>
            {
                result.push('\n');
                result.push_str(indent_str);
            }
            DefinitiveListTactic::Mixed => {
                let total_width = total_item_width(item) + item_sep_len;

                // 1 is space between separator and item.
                if (line_len > 0 && line_len + 1 + total_width > formatting.shape.width)
                    || prev_item_had_post_comment
                    || (formatting.nested
                        && (prev_item_is_nested_import || (!first && inner_item.contains("::"))))
                {
                    result.push('\n');
                    result.push_str(indent_str);
                    line_len = 0;
                    if formatting.ends_with_newline {
                        trailing_separator = true;
                    }
                } else if line_len > 0 {
                    result.push(' ');
                    line_len += 1;
                }

                if last && formatting.ends_with_newline {
                    separate = formatting.trailing_separator != SeparatorTactic::Never;
                }

                line_len += total_width;
            }
            _ => {}
        }

        // Pre-comments
        if let Some(ref comment) = item.pre_comment {
            // Block style in non-vertical mode.
            let block_mode = tactic == DefinitiveListTactic::Horizontal;
            // Width restriction is only relevant in vertical mode.
            let comment =
                rewrite_comment(comment, block_mode, formatting.shape, formatting.config)?;
            result.push_str(&comment);

            if !inner_item.is_empty() {
                use DefinitiveListTactic::*;
                if matches!(tactic, Vertical | Mixed | SpecialMacro(_)) {
                    // We cannot keep pre-comments on the same line if the comment is normalized.
                    let keep_comment = if formatting.config.normalize_comments()
                        || item.pre_comment_style == ListItemCommentStyle::DifferentLine
                    {
                        false
                    } else {
                        // We will try to keep the comment on the same line with the item here.
                        // 1 = ` `
                        let total_width = total_item_width(item) + item_sep_len + 1;
                        total_width <= formatting.shape.width
                    };
                    if keep_comment {
                        result.push(' ');
                    } else {
                        result.push('\n');
                        result.push_str(indent_str);
                        // This is the width of the item (without comments).
                        line_len = item.item.as_ref().map_or(0, |s| unicode_str_width(&s));
                    }
                } else {
                    result.push(' ')
                }
            }
            item_max_width = None;
        }

        if separate && sep_place.is_front() && !first {
            result.push_str(formatting.separator.trim());
            result.push(' ');
        }
        result.push_str(inner_item);

        // Post-comments
        if tactic == DefinitiveListTactic::Horizontal && item.post_comment.is_some() {
            let comment = item.post_comment.as_ref().unwrap();
            let formatted_comment = rewrite_comment(
                comment,
                true,
                Shape::legacy(formatting.shape.width, Indent::empty()),
                formatting.config,
            )?;

            result.push(' ');
            result.push_str(&formatted_comment);
        }

        if separate && sep_place.is_back() {
            result.push_str(formatting.separator);
        }

        if tactic != DefinitiveListTactic::Horizontal && item.post_comment.is_some() {
            let comment = item.post_comment.as_ref().unwrap();
            let overhead = last_line_width(&result) + first_line_width(comment.trim());

            let rewrite_post_comment = |item_max_width: &mut Option<usize>| {
                if item_max_width.is_none() && !last && !inner_item.contains('\n') {
                    *item_max_width = Some(max_width_of_item_with_post_comment(
                        &cloned_items,
                        i,
                        overhead,
                        formatting.config.max_width(),
                    ));
                }
                let overhead = if starts_with_newline(comment) {
                    0
                } else if let Some(max_width) = *item_max_width {
                    max_width + 2
                } else {
                    // 1 = space between item and comment.
                    item_last_line_width + 1
                };
                let width = formatting.shape.width.checked_sub(overhead).unwrap_or(1);
                let offset = formatting.shape.indent + overhead;
                let comment_shape = Shape::legacy(width, offset);

                // Use block-style only for the last item or multiline comments.
                let block_style = !formatting.ends_with_newline && last
                    || comment.trim().contains('\n')
                    || comment.trim().len() > width;

                rewrite_comment(
                    comment.trim_start(),
                    block_style,
                    comment_shape,
                    formatting.config,
                )
            };

            let mut formatted_comment = rewrite_post_comment(&mut item_max_width)?;

            if !starts_with_newline(comment) {
                if formatting.align_comments {
                    let mut comment_alignment =
                        post_comment_alignment(item_max_width, inner_item.len());
                    if first_line_width(&formatted_comment)
                        + last_line_width(&result)
                        + comment_alignment
                        + 1
                        > formatting.config.max_width()
                    {
                        item_max_width = None;
                        formatted_comment = rewrite_post_comment(&mut item_max_width)?;
                        comment_alignment =
                            post_comment_alignment(item_max_width, inner_item.len());
                    }
                    for _ in 0..=comment_alignment {
                        result.push(' ');
                    }
                }
                // An additional space for the missing trailing separator (or
                // if we skipped alignment above).
                if !formatting.align_comments
                    || (last
                        && item_max_width.is_some()
                        && !separate
                        && !formatting.separator.is_empty())
                {
                    result.push(' ');
                }
            } else {
                result.push('\n');
                result.push_str(indent_str);
            }
            if formatted_comment.contains('\n') {
                item_max_width = None;
            }
            result.push_str(&formatted_comment);
        } else {
            item_max_width = None;
        }

        if formatting.preserve_newline
            && !last
            && tactic == DefinitiveListTactic::Vertical
            && item.new_lines
        {
            item_max_width = None;
            result.push('\n');
        }

        prev_item_had_post_comment = item.post_comment.is_some();
        prev_item_is_nested_import = inner_item.contains("::");
    }

    Some(result)
}

fn max_width_of_item_with_post_comment<I, T>(
    items: &I,
    i: usize,
    overhead: usize,
    max_budget: usize,
) -> usize
where
    I: IntoIterator<Item = T> + Clone,
    T: AsRef<ListItem>,
{
    let mut max_width = 0;
    let mut first = true;
    for item in items.clone().into_iter().skip(i) {
        let item = item.as_ref();
        let inner_item_width = item.inner_as_ref().len();
        if !first
            && (item.is_different_group()
                || item.post_comment.is_none()
                || inner_item_width + overhead > max_budget)
        {
            return max_width;
        }
        if max_width < inner_item_width {
            max_width = inner_item_width;
        }
        if item.new_lines {
            return max_width;
        }
        first = false;
    }
    max_width
}

fn post_comment_alignment(item_max_width: Option<usize>, inner_item_len: usize) -> usize {
    item_max_width.unwrap_or(0).saturating_sub(inner_item_len)
}

pub(crate) struct ListItems<'a, I, F1, F2, F3>
where
    I: Iterator,
{
    snippet_provider: &'a SnippetProvider,
    inner: Peekable<I>,
    get_lo: F1,
    get_hi: F2,
    get_item_string: F3,
    prev_span_end: BytePos,
    next_span_start: BytePos,
    terminator: &'a str,
    separator: &'a str,
    leave_last: bool,
}

pub(crate) fn extract_pre_comment(pre_snippet: &str) -> (Option<String>, ListItemCommentStyle) {
    let trimmed_pre_snippet = pre_snippet.trim();
    // Both start and end are checked to support keeping a block comment inline with
    // the item, even if there are preceeding line comments, while still supporting
    // a snippet that starts with a block comment but also contains one or more
    // trailing single line comments.
    // https://github.com/rust-lang/rustfmt/issues/3025
    // https://github.com/rust-lang/rustfmt/pull/3048
    // https://github.com/rust-lang/rustfmt/issues/3839
    let starts_with_block_comment = trimmed_pre_snippet.starts_with("/*");
    let ends_with_block_comment = trimmed_pre_snippet.ends_with("*/");
    let starts_with_single_line_comment = trimmed_pre_snippet.starts_with("//");
    if ends_with_block_comment {
        let comment_end = pre_snippet.rfind(|c| c == '/').unwrap();
        if pre_snippet[comment_end..].contains('\n') {
            (
                Some(trimmed_pre_snippet.to_owned()),
                ListItemCommentStyle::DifferentLine,
            )
        } else {
            (
                Some(trimmed_pre_snippet.to_owned()),
                ListItemCommentStyle::SameLine,
            )
        }
    } else if starts_with_single_line_comment || starts_with_block_comment {
        (
            Some(trimmed_pre_snippet.to_owned()),
            ListItemCommentStyle::DifferentLine,
        )
    } else {
        (None, ListItemCommentStyle::None)
    }
}

pub(crate) fn extract_post_comment(
    post_snippet: &str,
    comment_end: usize,
    separator: &str,
) -> Option<String> {
    let white_space: &[_] = &[' ', '\t'];

    // Cleanup post-comment: strip separators and whitespace.
    let post_snippet = post_snippet[..comment_end].trim();
    let post_snippet_trimmed = if post_snippet.starts_with(|c| c == ',' || c == ':') {
        post_snippet[1..].trim_matches(white_space)
    } else if let Some(stripped) = post_snippet.strip_prefix(separator) {
        stripped.trim_matches(white_space)
    }
    // not comment or over two lines
    else if post_snippet.ends_with(',')
        && (!post_snippet.trim().starts_with("//") || post_snippet.trim().contains('\n'))
    {
        post_snippet[..(post_snippet.len() - 1)].trim_matches(white_space)
    } else {
        post_snippet
    };
    // FIXME(#3441): post_snippet includes 'const' now
    // it should not include here
    let removed_newline_snippet = post_snippet_trimmed.trim();
    if !post_snippet_trimmed.is_empty()
        && (removed_newline_snippet.starts_with("//") || removed_newline_snippet.starts_with("/*"))
    {
        Some(post_snippet_trimmed.to_owned())
    } else {
        None
    }
}

pub(crate) fn get_comment_end(
    post_snippet: &str,
    separator: &str,
    terminator: &str,
    is_last: bool,
) -> usize {
    if is_last {
        return post_snippet
            .find_uncommented(terminator)
            .unwrap_or_else(|| post_snippet.len());
    }

    let mut block_open_index = post_snippet.find("/*");
    // check if it really is a block comment (and not `//*` or a nested comment)
    if let Some(i) = block_open_index {
        match post_snippet.find('/') {
            Some(j) if j < i => block_open_index = None,
            _ if post_snippet[..i].ends_with('/') => block_open_index = None,
            _ => (),
        }
    }
    let newline_index = post_snippet.find('\n');
    if let Some(separator_index) = post_snippet.find_uncommented(separator) {
        match (block_open_index, newline_index) {
            // Separator before comment, with the next item on same line.
            // Comment belongs to next item.
            (Some(i), None) if i > separator_index => separator_index + 1,
            // Block-style post-comment before the separator.
            (Some(i), None) => cmp::max(
                find_comment_end(&post_snippet[i..]).unwrap() + i,
                separator_index + 1,
            ),
            // Block-style post-comment. Either before or after the separator.
            (Some(i), Some(j)) if i < j => cmp::max(
                find_comment_end(&post_snippet[i..]).unwrap() + i,
                separator_index + 1,
            ),
            // Potential *single* line comment.
            (_, Some(j)) if j > separator_index => j + 1,
            _ => post_snippet.len(),
        }
    } else if let Some(newline_index) = newline_index {
        // Match arms may not have trailing comma. In any case, for match arms,
        // we will assume that the post comment belongs to the next arm if they
        // do not end with trailing comma.
        newline_index + 1
    } else {
        0
    }
}

// Account for extra whitespace between items. This is fiddly
// because of the way we divide pre- and post- comments.
pub(crate) fn has_extra_newline(post_snippet: &str, comment_end: usize) -> bool {
    if post_snippet.is_empty() || comment_end == 0 {
        return false;
    }

    let len_last = post_snippet[..comment_end]
        .chars()
        .last()
        .unwrap()
        .len_utf8();
    // Everything from the separator to the next item.
    let test_snippet = &post_snippet[comment_end - len_last..];
    let first_newline = test_snippet
        .find('\n')
        .unwrap_or_else(|| test_snippet.len());
    // From the end of the first line of comments.
    let test_snippet = &test_snippet[first_newline..];
    let first = test_snippet
        .find(|c: char| !c.is_whitespace())
        .unwrap_or_else(|| test_snippet.len());
    // From the end of the first line of comments to the next non-whitespace char.
    let test_snippet = &test_snippet[..first];

    // There were multiple line breaks which got trimmed to nothing.
    count_newlines(test_snippet) > 1
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
        self.inner.next().map(|item| {
            // Pre-comment
            let pre_snippet = self
                .snippet_provider
                .span_to_snippet(mk_sp(self.prev_span_end, (self.get_lo)(&item)))
                .unwrap_or("");
            let (pre_comment, pre_comment_style) = extract_pre_comment(pre_snippet);

            // Post-comment
            let next_start = match self.inner.peek() {
                Some(next_item) => (self.get_lo)(next_item),
                None => self.next_span_start,
            };
            let post_snippet = self
                .snippet_provider
                .span_to_snippet(mk_sp((self.get_hi)(&item), next_start))
                .unwrap_or("");
            let comment_end = get_comment_end(
                post_snippet,
                self.separator,
                self.terminator,
                self.inner.peek().is_none(),
            );
            let new_lines = has_extra_newline(post_snippet, comment_end);
            let post_comment = extract_post_comment(post_snippet, comment_end, self.separator);

            self.prev_span_end = (self.get_hi)(&item) + BytePos(comment_end as u32);

            ListItem {
                pre_comment,
                pre_comment_style,
                item: if self.inner.peek().is_none() && self.leave_last {
                    None
                } else {
                    (self.get_item_string)(&item)
                },
                post_comment,
                new_lines,
            }
        })
    }
}

#[allow(clippy::too_many_arguments)]
// Creates an iterator over a list's items with associated comments.
pub(crate) fn itemize_list<'a, T, I, F1, F2, F3>(
    snippet_provider: &'a SnippetProvider,
    inner: I,
    terminator: &'a str,
    separator: &'a str,
    get_lo: F1,
    get_hi: F2,
    get_item_string: F3,
    prev_span_end: BytePos,
    next_span_start: BytePos,
    leave_last: bool,
) -> ListItems<'a, I, F1, F2, F3>
where
    I: Iterator<Item = T>,
    F1: Fn(&T) -> BytePos,
    F2: Fn(&T) -> BytePos,
    F3: Fn(&T) -> Option<String>,
{
    ListItems {
        snippet_provider,
        inner: inner.peekable(),
        get_lo,
        get_hi,
        get_item_string,
        prev_span_end,
        next_span_start,
        terminator,
        separator,
        leave_last,
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

pub(crate) fn total_item_width(item: &ListItem) -> usize {
    comment_len(item.pre_comment.as_ref().map(|x| &(*x)[..]))
        + comment_len(item.post_comment.as_ref().map(|x| &(*x)[..]))
        + item.item.as_ref().map_or(0, |s| unicode_str_width(&s))
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
pub(crate) fn struct_lit_shape(
    shape: Shape,
    context: &RewriteContext<'_>,
    prefix_width: usize,
    suffix_width: usize,
) -> Option<(Option<Shape>, Shape)> {
    let v_shape = match context.config.indent_style() {
        IndentStyle::Visual => shape
            .visual_indent(0)
            .shrink_left(prefix_width)?
            .sub_width(suffix_width)?,
        IndentStyle::Block => {
            let shape = shape.block_indent(context.config.tab_spaces());
            Shape {
                width: context.budget(shape.indent.width()),
                ..shape
            }
        }
    };
    let shape_width = shape.width.checked_sub(prefix_width + suffix_width);
    if let Some(w) = shape_width {
        let shape_width = cmp::min(w, context.config.struct_lit_width());
        Some((Some(Shape::legacy(shape_width, shape.indent)), v_shape))
    } else {
        Some((None, v_shape))
    }
}

// Compute the tactic for the internals of a struct-lit-like thing.
pub(crate) fn struct_lit_tactic(
    h_shape: Option<Shape>,
    context: &RewriteContext<'_>,
    items: &[ListItem],
) -> DefinitiveListTactic {
    if let Some(h_shape) = h_shape {
        let prelim_tactic = match (context.config.indent_style(), items.len()) {
            (IndentStyle::Visual, 1) => ListTactic::HorizontalVertical,
            _ if context.config.struct_lit_single_line() => ListTactic::HorizontalVertical,
            _ => ListTactic::Vertical,
        };
        definitive_tactic(items, prelim_tactic, Separator::Comma, h_shape.width)
    } else {
        DefinitiveListTactic::Vertical
    }
}

// Given a tactic and possible shapes for horizontal and vertical layout,
// come up with the actual shape to use.
pub(crate) fn shape_for_tactic(
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
pub(crate) fn struct_lit_formatting<'a>(
    shape: Shape,
    tactic: DefinitiveListTactic,
    context: &'a RewriteContext<'_>,
    force_no_trailing_comma: bool,
) -> ListFormatting<'a> {
    let ends_with_newline = context.config.indent_style() != IndentStyle::Visual
        && tactic == DefinitiveListTactic::Vertical;
    ListFormatting {
        tactic,
        separator: ",",
        trailing_separator: if force_no_trailing_comma {
            SeparatorTactic::Never
        } else {
            context.config.trailing_comma()
        },
        separator_place: SeparatorPlace::Back,
        shape,
        ends_with_newline,
        preserve_newline: true,
        nested: false,
        align_comments: true,
        config: context.config,
    }
}
