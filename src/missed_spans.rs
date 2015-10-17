// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use visitor::FmtVisitor;

use syntax::codemap::{self, BytePos};

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
        debug!("format_missing_inner: {:?} to {:?}",
               self.codemap.lookup_char_pos(start),
               self.codemap.lookup_char_pos(end));

        if start == end {
            // Do nothing if this is the beginning of the file.
            if start == self.codemap.lookup_char_pos(start).file.start_pos {
                return;
            }
            process_last_snippet(self, "", "");
            return;
        }

        assert!(start < end,
                "Request to format inverted span: {:?} to {:?}",
                self.codemap.lookup_char_pos(start),
                self.codemap.lookup_char_pos(end));

        self.last_pos = end;
        let span = codemap::mk_sp(start, end);
        let snippet = self.snippet(span);

        self.write_snippet(&snippet, &process_last_snippet);
    }

    fn write_snippet<F: Fn(&mut FmtVisitor, &str, &str)>(&mut self,
                                                         snippet: &str,
                                                         process_last_snippet: F) {
        let mut lines: Vec<&str> = snippet.lines().collect();
        let last_snippet = if snippet.ends_with("\n") {
            ""
        } else {
            lines.pop().unwrap()
        };
        for line in lines.iter() {
            self.buffer.push_str(line.trim_right());
            self.buffer.push_str("\n");
        }
        process_last_snippet(self, &last_snippet, snippet);
    }
}
