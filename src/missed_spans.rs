// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use utils::make_indent;
use visitor::FmtVisitor;

use syntax::codemap::{self, BytePos};

impl<'a> FmtVisitor<'a> {
    // TODO these format_missing methods are ugly. Refactor and add unit tests
    // for the central whitespace stripping loop.
    pub fn format_missing(&mut self, end: BytePos) {
        self.format_missing_inner(end, |this, last_snippet, file_name, _| {
            this.changes.push_str(file_name, last_snippet)
        })
    }

    pub fn format_missing_with_indent(&mut self, end: BytePos) {
        self.format_missing_inner(end, |this, last_snippet, file_name, snippet| {
            this.changes.push_str(file_name, last_snippet.trim_right());
            if last_snippet == snippet {
                // No new lines in the snippet.
                this.changes.push_str(file_name, "\n");
            }
            let indent = make_indent(this.block_indent);
            this.changes.push_str(file_name, &indent);
        })
    }

    fn format_missing_inner<F: Fn(&mut FmtVisitor, &str, &str, &str)>(&mut self,
                                                                      end: BytePos,
                                                                      process_last_snippet: F)
    {
        let start = self.last_pos;
        debug!("format_missing_inner: {:?} to {:?}",
               self.codemap.lookup_char_pos(start),
               self.codemap.lookup_char_pos(end));

        if start == end {
            return;
        }

        assert!(start < end,
                "Request to format inverted span: {:?} to {:?}",
                self.codemap.lookup_char_pos(start),
                self.codemap.lookup_char_pos(end));

        self.last_pos = end;
        let spans = self.changes.filespans_for_span(start, end);
        for (i, &(start, end)) in spans.iter().enumerate() {
            let span = codemap::mk_sp(BytePos(start), BytePos(end));
            let file_name = &self.codemap.span_to_filename(span);
            let snippet = self.snippet(span);

            self.write_snippet(&snippet,
                               file_name,
                               i == spans.len() - 1,
                               &process_last_snippet);
        }
    }

    fn write_snippet<F: Fn(&mut FmtVisitor, &str, &str, &str)>(&mut self,
                                                               snippet: &str,
                                                               file_name: &str,
                                                               last_snippet: bool,
                                                               process_last_snippet: F) {
        // Trim whitespace from the right hand side of each line.
        // Annoyingly, the library functions for splitting by lines etc. are not
        // quite right, so we must do it ourselves.
        let mut line_start = 0;
        let mut last_wspace = None;
        for (i, c) in snippet.char_indices() {
            if c == '\n' {
                if let Some(lw) = last_wspace {
                    self.changes.push_str(file_name, &snippet[line_start..lw]);
                    self.changes.push_str(file_name, "\n");
                } else {
                    self.changes.push_str(file_name, &snippet[line_start..i+1]);
                }

                line_start = i + 1;
                last_wspace = None;
            } else {
                if c.is_whitespace() {
                    if last_wspace.is_none() {
                        last_wspace = Some(i);
                    }
                } else {
                    last_wspace = None;
                }
            }
        }
        if last_snippet {
            process_last_snippet(self, &snippet[line_start..], file_name, snippet);
        } else {
            self.changes.push_str(file_name, &snippet[line_start..]);
        }
    }
}
