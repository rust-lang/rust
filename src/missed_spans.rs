// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use WriteMode;
use visitor::FmtVisitor;
use syntax::codemap::{self, BytePos, Span, Pos};
use comment::{CodeCharKind, CommentCodeSlices, rewrite_comment};

impl<'a> FmtVisitor<'a> {
    // TODO these format_missing methods are ugly. Refactor and add unit tests
    // for the central whitespace stripping loop.
    pub fn format_missing(&mut self, end: BytePos) {
        self.format_missing_inner(end,
                                  |this, last_snippet, _| this.buffer.push_str(last_snippet))
    }

    pub fn format_missing_with_indent(&mut self, end: BytePos) {
        let config = self.config;
        self.format_missing_inner(end, |this, last_snippet, snippet| {
            this.buffer.push_str(last_snippet.trim_right());
            if last_snippet == snippet {
                // No new lines in the snippet.
                this.buffer.push_str("\n");
            }
            let indent = this.block_indent.to_string(config);
            this.buffer.push_str(&indent);
        })
    }

    fn format_missing_inner<F: Fn(&mut FmtVisitor, &str, &str)>(&mut self,
                                                                end: BytePos,
                                                                process_last_snippet: F) {
        let start = self.last_pos;

        if start == end {
            // Do nothing if this is the beginning of the file.
            if start != self.codemap.lookup_char_pos(start).file.start_pos {
                process_last_snippet(self, "", "");
            }
            return;
        }

        assert!(start < end,
                "Request to format inverted span: {:?} to {:?}",
                self.codemap.lookup_char_pos(start),
                self.codemap.lookup_char_pos(end));

        self.last_pos = end;
        let span = codemap::mk_sp(start, end);

        self.write_snippet(span, &process_last_snippet);
    }

    fn write_snippet<F>(&mut self, span: Span, process_last_snippet: F)
        where F: Fn(&mut FmtVisitor, &str, &str)
    {
        // Get a snippet from the file start to the span's hi without allocating.
        // We need it to determine what precedes the current comment. If the comment
        // follows code on the same line, we won't touch it.
        let big_span_lo = self.codemap.lookup_char_pos(span.lo).file.start_pos;
        let local_begin = self.codemap.lookup_byte_offset(big_span_lo);
        let local_end = self.codemap.lookup_byte_offset(span.hi);
        let start_index = local_begin.pos.to_usize();
        let end_index = local_end.pos.to_usize();
        let big_snippet = &local_begin.fm.src.as_ref().unwrap()[start_index..end_index];

        let big_diff = (span.lo - big_span_lo).to_usize();
        let snippet = self.snippet(span);

        self.write_snippet_inner(big_snippet, big_diff, &snippet, process_last_snippet);
    }

    fn write_snippet_inner<F>(&mut self,
                              big_snippet: &str,
                              big_diff: usize,
                              old_snippet: &str,
                              process_last_snippet: F)
        where F: Fn(&mut FmtVisitor, &str, &str)
    {
        // Trim whitespace from the right hand side of each line.
        // Annoyingly, the library functions for splitting by lines etc. are not
        // quite right, so we must do it ourselves.
        let mut line_start = 0;
        let mut last_wspace = None;
        let mut rewrite_next_comment = true;

        fn replace_chars(string: &str) -> String {
            string.chars()
                  .map(|ch| {
                      match ch.is_whitespace() {
                          true => ch,
                          false => 'X',
                      }
                  })
                  .collect()
        }

        let replaced = match self.write_mode {
            Some(mode) => {
                match mode {
                    WriteMode::Coverage => replace_chars(old_snippet),
                    _ => old_snippet.to_owned(),
                }
            }
            None => old_snippet.to_owned(),
        };
        let snippet = &*replaced;

        for (kind, offset, subslice) in CommentCodeSlices::new(snippet) {
            if let CodeCharKind::Comment = kind {
                let last_char = big_snippet[..(offset + big_diff)]
                                    .chars()
                                    .rev()
                                    .skip_while(|rev_c| [' ', '\t'].contains(&rev_c))
                                    .next();

                let fix_indent = last_char.map(|rev_c| ['{', '\n'].contains(&rev_c))
                                          .unwrap_or(true);

                if rewrite_next_comment && fix_indent {
                    if let Some('{') = last_char {
                        self.buffer.push_str("\n");
                    }

                    let comment_width = ::std::cmp::min(self.config.ideal_width,
                                                        self.config.max_width -
                                                        self.block_indent.width());

                    self.buffer.push_str(&self.block_indent.to_string(self.config));
                    self.buffer.push_str(&rewrite_comment(subslice,
                                                          false,
                                                          comment_width,
                                                          self.block_indent,
                                                          self.config)
                                              .unwrap());

                    last_wspace = None;
                    line_start = offset + subslice.len();

                    if let Some('/') = subslice.chars().skip(1).next() {
                        // Add a newline after line comments
                        self.buffer.push_str("\n");
                    } else if line_start < snippet.len() {
                        // For other comments add a newline if there isn't one at the end already
                        let c = snippet[line_start..].chars().next().unwrap();
                        if c != '\n' && c != '\r' {
                            self.buffer.push_str("\n");
                        }
                    }

                    continue;
                } else {
                    rewrite_next_comment = false;
                }
            }

            for (mut i, c) in subslice.char_indices() {
                i += offset;

                if c == '\n' {
                    if let Some(lw) = last_wspace {
                        self.buffer.push_str(&snippet[line_start..lw]);
                        self.buffer.push_str("\n");
                    } else {
                        self.buffer.push_str(&snippet[line_start..i + 1]);
                    }

                    line_start = i + 1;
                    last_wspace = None;
                    rewrite_next_comment = rewrite_next_comment || kind == CodeCharKind::Normal;
                } else {
                    if c.is_whitespace() {
                        if last_wspace.is_none() {
                            last_wspace = Some(i);
                        }
                    } else {
                        rewrite_next_comment = rewrite_next_comment || kind == CodeCharKind::Normal;
                        last_wspace = None;
                    }
                }
            }
        }

        process_last_snippet(self, &snippet[line_start..], &snippet);
    }
}
