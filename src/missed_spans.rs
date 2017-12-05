// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::borrow::Cow;
use std::iter::repeat;

use syntax::codemap::{BytePos, Pos, Span};

use codemap::LineRangeUtils;
use comment::{rewrite_comment, CodeCharKind, CommentCodeSlices};
use config::WriteMode;
use shape::{Indent, Shape};
use utils::{count_newlines, mk_sp};
use visitor::FmtVisitor;

impl<'a> FmtVisitor<'a> {
    fn output_at_start(&self) -> bool {
        self.buffer.len == 0
    }

    // TODO these format_missing methods are ugly. Refactor and add unit tests
    // for the central whitespace stripping loop.
    pub fn format_missing(&mut self, end: BytePos) {
        self.format_missing_inner(end, |this, last_snippet, _| {
            this.buffer.push_str(last_snippet)
        })
    }

    pub fn format_missing_with_indent(&mut self, end: BytePos) {
        let config = self.config;
        self.format_missing_inner(end, |this, last_snippet, snippet| {
            this.buffer.push_str(last_snippet.trim_right());
            if last_snippet == snippet && !this.output_at_start() {
                // No new lines in the snippet.
                this.buffer.push_str("\n");
            }
            let indent = this.block_indent.to_string(config);
            this.buffer.push_str(&indent);
        })
    }

    pub fn format_missing_no_indent(&mut self, end: BytePos) {
        self.format_missing_inner(end, |this, last_snippet, _| {
            this.buffer.push_str(last_snippet.trim_right());
        })
    }

    fn format_missing_inner<F: Fn(&mut FmtVisitor, &str, &str)>(
        &mut self,
        end: BytePos,
        process_last_snippet: F,
    ) {
        let start = self.last_pos;

        if start == end {
            // Do nothing if this is the beginning of the file.
            if !self.output_at_start() {
                process_last_snippet(self, "", "");
            }
            return;
        }

        assert!(
            start < end,
            "Request to format inverted span: {:?} to {:?}",
            self.codemap.lookup_char_pos(start),
            self.codemap.lookup_char_pos(end)
        );

        self.last_pos = end;
        let span = mk_sp(start, end);
        let snippet = self.snippet(span);
        if snippet.trim().is_empty() && !out_of_file_lines_range!(self, span) {
            // Keep vertical spaces within range.
            self.push_vertical_spaces(count_newlines(&snippet));
            process_last_snippet(self, "", &snippet);
        } else {
            self.write_snippet(span, &process_last_snippet);
        }
    }

    fn push_vertical_spaces(&mut self, mut newline_count: usize) {
        let newline_upper_bound = self.config.blank_lines_upper_bound() + 1;
        let newline_lower_bound = self.config.blank_lines_lower_bound() + 1;
        if newline_count > newline_upper_bound {
            newline_count = newline_upper_bound;
        } else if newline_count < newline_lower_bound {
            newline_count = newline_lower_bound;
        }
        let blank_lines: String = repeat('\n').take(newline_count).collect();
        self.buffer.push_str(&blank_lines);
    }

    fn write_snippet<F>(&mut self, span: Span, process_last_snippet: F)
    where
        F: Fn(&mut FmtVisitor, &str, &str),
    {
        // Get a snippet from the file start to the span's hi without allocating.
        // We need it to determine what precedes the current comment. If the comment
        // follows code on the same line, we won't touch it.
        let big_span_lo = self.codemap.lookup_char_pos(span.lo()).file.start_pos;
        let local_begin = self.codemap.lookup_byte_offset(big_span_lo);
        let local_end = self.codemap.lookup_byte_offset(span.hi());
        let start_index = local_begin.pos.to_usize();
        let end_index = local_end.pos.to_usize();
        let big_snippet = &local_begin.fm.src.as_ref().unwrap()[start_index..end_index];

        let big_diff = (span.lo() - big_span_lo).to_usize();
        let snippet = self.snippet(span);

        debug!("write_snippet `{}`", snippet);

        self.write_snippet_inner(big_snippet, big_diff, &snippet, span, process_last_snippet);
    }

    fn process_comment(
        &mut self,
        status: &mut SnippetStatus,
        snippet: &str,
        big_snippet: &str,
        offset: usize,
        big_diff: usize,
        subslice: &str,
        file_name: &str,
    ) -> bool {
        let last_char = big_snippet[..(offset + big_diff)]
            .chars()
            .rev()
            .skip_while(|rev_c| [' ', '\t'].contains(rev_c))
            .next();

        let fix_indent = last_char.map_or(true, |rev_c| ['{', '\n'].contains(&rev_c));

        let subslice_num_lines = count_newlines(subslice);
        let skip_this_range = !self.config.file_lines().intersects_range(
            file_name,
            status.cur_line,
            status.cur_line + subslice_num_lines,
        );

        if status.rewrite_next_comment && skip_this_range {
            status.rewrite_next_comment = false;
        }

        if status.rewrite_next_comment {
            if fix_indent {
                if let Some('{') = last_char {
                    self.buffer.push_str("\n");
                }
                self.buffer
                    .push_str(&self.block_indent.to_string(self.config));
            } else {
                self.buffer.push_str(" ");
            }

            let comment_width = ::std::cmp::min(
                self.config.comment_width(),
                self.config.max_width() - self.block_indent.width(),
            );
            let comment_indent = Indent::from_width(self.config, self.buffer.cur_offset());
            let comment_shape = Shape::legacy(comment_width, comment_indent);
            let comment_str = rewrite_comment(subslice, false, comment_shape, self.config)
                .unwrap_or_else(|| String::from(subslice));
            self.buffer.push_str(&comment_str);

            status.last_wspace = None;
            status.line_start = offset + subslice.len();

            if let Some('/') = subslice.chars().nth(1) {
                // check that there are no contained block comments
                if !subslice
                    .split('\n')
                    .map(|s| s.trim_left())
                    .any(|s| s.len() >= 2 && &s[0..2] == "/*")
                {
                    // Add a newline after line comments
                    self.buffer.push_str("\n");
                }
            } else if status.line_start <= snippet.len() {
                // For other comments add a newline if there isn't one at the end already
                match snippet[status.line_start..].chars().next() {
                    Some('\n') | Some('\r') => (),
                    _ => self.buffer.push_str("\n"),
                }
            }

            status.cur_line += subslice_num_lines;
            true
        } else {
            status.rewrite_next_comment = false;
            false
        }
    }

    fn write_snippet_inner<F>(
        &mut self,
        big_snippet: &str,
        big_diff: usize,
        old_snippet: &str,
        span: Span,
        process_last_snippet: F,
    ) where
        F: Fn(&mut FmtVisitor, &str, &str),
    {
        // Trim whitespace from the right hand side of each line.
        // Annoyingly, the library functions for splitting by lines etc. are not
        // quite right, so we must do it ourselves.
        let char_pos = self.codemap.lookup_char_pos(span.lo());
        let file_name = &char_pos.file.name;
        let mut status = SnippetStatus::new(char_pos.line);

        fn replace_chars<'a>(string: &'a str) -> Cow<'a, str> {
            if string.contains(char::is_whitespace) {
                Cow::from(
                    string
                        .chars()
                        .map(|ch| if ch.is_whitespace() { ch } else { 'X' })
                        .collect::<String>(),
                )
            } else {
                Cow::from(string)
            }
        }

        let snippet = &*match self.config.write_mode() {
            WriteMode::Coverage => replace_chars(old_snippet),
            _ => Cow::from(old_snippet),
        };

        for (kind, offset, subslice) in CommentCodeSlices::new(snippet) {
            debug!("{:?}: {:?}", kind, subslice);

            if let CodeCharKind::Comment = kind {
                if self.process_comment(
                    &mut status,
                    snippet,
                    big_snippet,
                    offset,
                    big_diff,
                    subslice,
                    file_name,
                ) {
                    continue;
                }
            }

            let newline_count = count_newlines(&subslice);
            if subslice.trim().is_empty() && newline_count > 0
                && self.config.file_lines().intersects_range(
                    file_name,
                    status.cur_line,
                    status.cur_line + newline_count,
                ) {
                self.push_vertical_spaces(newline_count);
                status.cur_line += newline_count;
                status.rewrite_next_comment = true;
                status.line_start = offset + newline_count;
            } else {
                for (mut i, c) in subslice.char_indices() {
                    i += offset;

                    if c == '\n' {
                        if !self.config
                            .file_lines()
                            .contains_line(file_name, status.cur_line)
                        {
                            status.last_wspace = None;
                        }

                        if let Some(lw) = status.last_wspace {
                            self.buffer.push_str(&snippet[status.line_start..lw]);
                            self.buffer.push_str("\n");
                        } else {
                            self.buffer.push_str(&snippet[status.line_start..i + 1]);
                        }

                        status.cur_line += 1;
                        status.line_start = i + 1;
                        status.last_wspace = None;
                        status.rewrite_next_comment = true;
                    } else if c.is_whitespace() {
                        if status.last_wspace.is_none() {
                            status.last_wspace = Some(i);
                        }
                    } else if c == ';' {
                        if status.last_wspace.is_some() {
                            status.line_start = i;
                        }

                        status.rewrite_next_comment = true;
                        status.last_wspace = None;
                    } else {
                        status.rewrite_next_comment = true;
                        status.last_wspace = None;
                    }
                }

                let remaining = snippet[status.line_start..subslice.len() + offset].trim();
                if !remaining.is_empty() {
                    self.buffer.push_str(remaining);
                    status.line_start = subslice.len() + offset;
                    status.rewrite_next_comment = true;
                }
            }
        }

        process_last_snippet(self, &snippet[status.line_start..], snippet);
    }
}

struct SnippetStatus {
    line_start: usize,
    last_wspace: Option<usize>,
    rewrite_next_comment: bool,
    cur_line: usize,
}

impl SnippetStatus {
    fn new(cur_line: usize) -> Self {
        SnippetStatus {
            line_start: 0,
            last_wspace: None,
            rewrite_next_comment: true,
            cur_line,
        }
    }
}
