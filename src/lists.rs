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

use syntax::codemap::{self, CodeMap, BytePos};

use Indent;
use utils::{round_up_to_power_of_two, wrap_str};
use comment::{FindUncommented, rewrite_comment, find_comment_end};
use config::Config;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum ListTactic {
    // One item per row.
    Vertical,
    // All items on one row.
    Horizontal,
    // Try Horizontal layout, if that fails then vertical
    HorizontalVertical,
    // Pack as many items as possible per row over (possibly) many rows.
    Mixed,
}

impl_enum_decodable!(ListTactic, Vertical, Horizontal, HorizontalVertical, Mixed);

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum SeparatorTactic {
    Always,
    Never,
    Vertical,
}

impl_enum_decodable!(SeparatorTactic, Always, Never, Vertical);

// TODO having some helpful ctors for ListFormatting would be nice.
pub struct ListFormatting<'a> {
    pub tactic: ListTactic,
    pub separator: &'a str,
    pub trailing_separator: SeparatorTactic,
    pub indent: Indent,
    // Available width if we layout horizontally.
    pub h_width: usize,
    // Available width if we layout vertically
    pub v_width: usize,
    // Non-expressions, e.g. items, will have a new line at the end of the list.
    // Important for comment styles.
    pub ends_with_newline: bool,
    pub config: &'a Config,
}

impl<'a> ListFormatting<'a> {
    pub fn for_fn(width: usize, offset: Indent, config: &'a Config) -> ListFormatting<'a> {
        ListFormatting {
            tactic: ListTactic::HorizontalVertical,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: offset,
            h_width: width,
            v_width: width,
            ends_with_newline: false,
            config: config,
        }
    }
}

pub struct ListItem {
    pub pre_comment: Option<String>,
    // Item should include attributes and doc comments.
    pub item: String,
    pub post_comment: Option<String>,
    // Whether there is extra whitespace before this item.
    pub new_lines: bool,
}

impl ListItem {
    pub fn is_multiline(&self) -> bool {
        self.item.contains('\n') || self.pre_comment.is_some() ||
        self.post_comment.as_ref().map(|s| s.contains('\n')).unwrap_or(false)
    }

    pub fn has_line_pre_comment(&self) -> bool {
        self.pre_comment.as_ref().map_or(false, |comment| comment.starts_with("//"))
    }

    pub fn from_str<S: Into<String>>(s: S) -> ListItem {
        ListItem { pre_comment: None, item: s.into(), post_comment: None, new_lines: false }
    }
}

// Format a list of commented items into a string.
// FIXME: this has grown into a monstrosity
// TODO: add unit tests
pub fn write_list<'b>(items: &[ListItem], formatting: &ListFormatting<'b>) -> Option<String> {
    if items.is_empty() {
        return Some(String::new());
    }

    let mut tactic = formatting.tactic;

    // Conservatively overestimates because of the changing separator tactic.
    let sep_count = if formatting.trailing_separator == SeparatorTactic::Always {
        items.len()
    } else {
        items.len() - 1
    };
    let sep_len = formatting.separator.len();
    let total_sep_len = (sep_len + 1) * sep_count;
    let total_width = calculate_width(items);
    let fits_single = total_width + total_sep_len <= formatting.h_width;

    // Check if we need to fallback from horizontal listing, if possible.
    if tactic == ListTactic::HorizontalVertical {
        debug!("write_list: total_width: {}, total_sep_len: {}, h_width: {}",
               total_width,
               total_sep_len,
               formatting.h_width);
        tactic = if fits_single && !items.iter().any(ListItem::is_multiline) {
            ListTactic::Horizontal
        } else {
            ListTactic::Vertical
        };
    }

    // Check if we can fit everything on a single line in mixed mode.
    // The horizontal tactic does not break after v_width columns.
    if tactic == ListTactic::Mixed && fits_single {
        tactic = ListTactic::Horizontal;
    }

    // Switch to vertical mode if we find non-block comments.
    if items.iter().any(ListItem::has_line_pre_comment) {
        tactic = ListTactic::Vertical;
    }

    // Now that we know how we will layout, we can decide for sure if there
    // will be a trailing separator.
    let trailing_separator = needs_trailing_separator(formatting.trailing_separator, tactic);

    // Create a buffer for the result.
    // TODO could use a StringBuffer or rope for this
    let alloc_width = if tactic == ListTactic::Horizontal {
        total_width + total_sep_len
    } else {
        total_width + items.len() * (formatting.indent.width() + 1)
    };
    let mut result = String::with_capacity(round_up_to_power_of_two(alloc_width));

    let mut line_len = 0;
    let indent_str = &formatting.indent.to_string(formatting.config);
    for (i, item) in items.iter().enumerate() {
        let first = i == 0;
        let last = i == items.len() - 1;
        let separate = !last || trailing_separator;
        let item_sep_len = if separate {
            sep_len
        } else {
            0
        };
        let item_width = item.item.len() + item_sep_len;

        match tactic {
            ListTactic::Horizontal if !first => {
                result.push(' ');
            }
            ListTactic::Vertical if !first => {
                result.push('\n');
                result.push_str(indent_str);
            }
            ListTactic::Mixed => {
                let total_width = total_item_width(item) + item_sep_len;

                if line_len > 0 && line_len + total_width > formatting.v_width {
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
            let block_mode = tactic != ListTactic::Vertical;
            // Width restriction is only relevant in vertical mode.
            let max_width = formatting.v_width;
            result.push_str(&rewrite_comment(comment,
                                             block_mode,
                                             max_width,
                                             formatting.indent,
                                             formatting.config));

            if tactic == ListTactic::Vertical {
                result.push('\n');
                result.push_str(indent_str);
            } else {
                result.push(' ');
            }
        }

        let max_width = formatting.indent.width() + formatting.v_width;
        let item_str = wrap_str(&item.item[..], max_width, formatting.v_width, formatting.indent);
        result.push_str(&&try_opt!(item_str));

        // Post-comments
        if tactic != ListTactic::Vertical && item.post_comment.is_some() {
            let comment = item.post_comment.as_ref().unwrap();
            let formatted_comment = rewrite_comment(comment,
                                                    true,
                                                    formatting.v_width,
                                                    Indent::empty(),
                                                    formatting.config);

            result.push(' ');
            result.push_str(&formatted_comment);
        }

        if separate {
            result.push_str(formatting.separator);
        }

        if tactic == ListTactic::Vertical && item.post_comment.is_some() {
            // 1 = space between item and comment.
            let width = formatting.v_width.checked_sub(item_width + 1).unwrap_or(1);
            let mut offset = formatting.indent;
            offset.alignment += item_width + 1;
            let comment = item.post_comment.as_ref().unwrap();
            // Use block-style only for the last item or multiline comments.
            let block_style = !formatting.ends_with_newline && last ||
                              comment.trim().contains('\n') ||
                              comment.trim().len() > width;

            let formatted_comment = rewrite_comment(comment,
                                                    block_style,
                                                    width,
                                                    offset,
                                                    formatting.config);

            result.push(' ');
            result.push_str(&formatted_comment);
        }

        if !last && tactic == ListTactic::Vertical && item.new_lines {
            result.push('\n');
        }
    }

    Some(result)
}

pub struct ListItems<'a, I, F1, F2, F3>
    where I: Iterator
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
    where I: Iterator<Item = T>,
          F1: Fn(&T) -> BytePos,
          F2: Fn(&T) -> BytePos,
          F3: Fn(&T) -> String
{
    type Item = ListItem;

    fn next(&mut self) -> Option<Self::Item> {
        let white_space: &[_] = &[' ', '\t'];

        self.inner.next().map(|item| {
            let mut new_lines = false;
            // Pre-comment
            let pre_snippet = self.codemap
                                  .span_to_snippet(codemap::mk_sp(self.prev_span_end,
                                                                  (self.get_lo)(&item)))
                                  .unwrap();
            let trimmed_pre_snippet = pre_snippet.trim();
            let pre_comment = if !trimmed_pre_snippet.is_empty() {
                Some(trimmed_pre_snippet.to_owned())
            } else {
                None
            };

            // Post-comment
            let next_start = match self.inner.peek() {
                Some(ref next_item) => (self.get_lo)(next_item),
                None => self.next_span_start,
            };
            let post_snippet = self.codemap
                                   .span_to_snippet(codemap::mk_sp((self.get_hi)(&item),
                                                                   next_start))
                                   .unwrap();

            let comment_end = match self.inner.peek() {
                Some(..) => {
                    let block_open_index = post_snippet.find("/*");
                    let newline_index = post_snippet.find('\n');
                    let separator_index = post_snippet.find_uncommented(",").unwrap();

                    match (block_open_index, newline_index) {
                        // Separator before comment, with the next item on same line.
                        // Comment belongs to next item.
                        (Some(i), None) if i > separator_index => {
                            separator_index + 1
                        }
                        // Block-style post-comment before the separator.
                        (Some(i), None) => {
                            cmp::max(find_comment_end(&post_snippet[i..]).unwrap() + i,
                                     separator_index + 1)
                        }
                        // Block-style post-comment. Either before or after the separator.
                        (Some(i), Some(j)) if i < j => {
                            cmp::max(find_comment_end(&post_snippet[i..]).unwrap() + i,
                                     separator_index + 1)
                        }
                        // Potential *single* line comment.
                        (_, Some(j)) => j + 1,
                        _ => post_snippet.len(),
                    }
                }
                None => {
                    post_snippet.find_uncommented(self.terminator).unwrap_or(post_snippet.len())
                }
            };

            if !post_snippet.is_empty() && comment_end > 0 {
                // Account for extra whitespace between items. This is fiddly
                // because of the way we divide pre- and post- comments.

                // Everything from the separator to the next item.
                let test_snippet = &post_snippet[comment_end-1..];
                let first_newline = test_snippet.find('\n').unwrap_or(test_snippet.len());
                // From the end of the first line of comments.
                let test_snippet = &test_snippet[first_newline..];
                let first = test_snippet.find(|c: char| !c.is_whitespace())
                                        .unwrap_or(test_snippet.len());
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
            } else if post_snippet.ends_with(",") {
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
pub fn itemize_list<'a, T, I, F1, F2, F3>(codemap: &'a CodeMap,
                                          inner: I,
                                          terminator: &'a str,
                                          get_lo: F1,
                                          get_hi: F2,
                                          get_item_string: F3,
                                          prev_span_end: BytePos,
                                          next_span_start: BytePos)
                                          -> ListItems<'a, I, F1, F2, F3>
    where I: Iterator<Item = T>,
          F1: Fn(&T) -> BytePos,
          F2: Fn(&T) -> BytePos,
          F3: Fn(&T) -> String
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

fn needs_trailing_separator(separator_tactic: SeparatorTactic, list_tactic: ListTactic) -> bool {
    match separator_tactic {
        SeparatorTactic::Always => true,
        SeparatorTactic::Vertical => list_tactic == ListTactic::Vertical,
        SeparatorTactic::Never => false,
    }
}

fn calculate_width(items: &[ListItem]) -> usize {
    items.iter().map(total_item_width).fold(0, |a, l| a + l)
}

fn total_item_width(item: &ListItem) -> usize {
    comment_len(&item.pre_comment) + comment_len(&item.post_comment) + item.item.len()
}

fn comment_len(comment: &Option<String>) -> usize {
    match *comment {
        Some(ref s) => {
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
