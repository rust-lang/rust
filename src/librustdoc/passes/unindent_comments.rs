// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp;
use std::string::String;
use std::usize;

use clean::{self, Item};
use plugins;
use fold::{self, DocFolder};

pub fn unindent_comments(krate: clean::Crate) -> plugins::PluginResult {
    CommentCleaner.fold_crate(krate)
}

struct CommentCleaner;

impl fold::DocFolder for CommentCleaner {
    fn fold_item(&mut self, mut i: Item) -> Option<Item> {
        i.attrs.unindent_doc_comments();
        self.fold_item_recur(i)
    }
}

impl clean::Attributes {
    pub fn unindent_doc_comments(&mut self) {
        for doc_string in &mut self.doc_strings {
            *doc_string = unindent(doc_string);
        }
    }
}

fn unindent(s: &str) -> String {
    let lines = s.lines().collect::<Vec<&str> >();
    let mut saw_first_line = false;
    let mut saw_second_line = false;
    let min_indent = lines.iter().fold(usize::MAX, |min_indent, line| {

        // After we see the first non-whitespace line, look at
        // the line we have. If it is not whitespace, and therefore
        // part of the first paragraph, then ignore the indentation
        // level of the first line
        let ignore_previous_indents =
            saw_first_line &&
            !saw_second_line &&
            !line.chars().all(|c| c.is_whitespace());

        let min_indent = if ignore_previous_indents {
            usize::MAX
        } else {
            min_indent
        };

        if saw_first_line {
            saw_second_line = true;
        }

        if line.chars().all(|c| c.is_whitespace()) {
            min_indent
        } else {
            saw_first_line = true;
            let mut whitespace = 0;
            line.chars().all(|char| {
                // Compare against either space or tab, ignoring whether they
                // are mixed or not
                if char == ' ' || char == '\t' {
                    whitespace += 1;
                    true
                } else {
                    false
                }
            });
            cmp::min(min_indent, whitespace)
        }
    });

    if !lines.is_empty() {
        let mut unindented = vec![ lines[0].trim().to_string() ];
        unindented.extend_from_slice(&lines[1..].iter().map(|&line| {
            if line.chars().all(|c| c.is_whitespace()) {
                line.to_string()
            } else {
                assert!(line.len() >= min_indent);
                line[min_indent..].to_string()
            }
        }).collect::<Vec<_>>());
        unindented.join("\n")
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod unindent_tests {
    use super::unindent;

    #[test]
    fn should_unindent() {
        let s = "    line1\n    line2".to_string();
        let r = unindent(&s);
        assert_eq!(r, "line1\nline2");
    }

    #[test]
    fn should_unindent_multiple_paragraphs() {
        let s = "    line1\n\n    line2".to_string();
        let r = unindent(&s);
        assert_eq!(r, "line1\n\nline2");
    }

    #[test]
    fn should_leave_multiple_indent_levels() {
        // Line 2 is indented another level beyond the
        // base indentation and should be preserved
        let s = "    line1\n\n        line2".to_string();
        let r = unindent(&s);
        assert_eq!(r, "line1\n\n    line2");
    }

    #[test]
    fn should_ignore_first_line_indent() {
        // The first line of the first paragraph may not be indented as
        // far due to the way the doc string was written:
        //
        // #[doc = "Start way over here
        //          and continue here"]
        let s = "line1\n    line2".to_string();
        let r = unindent(&s);
        assert_eq!(r, "line1\nline2");
    }

    #[test]
    fn should_not_ignore_first_line_indent_in_a_single_line_para() {
        let s = "line1\n\n    line2".to_string();
        let r = unindent(&s);
        assert_eq!(r, "line1\n\n    line2");
    }

    #[test]
    fn should_unindent_tabs() {
        let s = "\tline1\n\tline2".to_string();
        let r = unindent(&s);
        assert_eq!(r, "line1\nline2");
    }

    #[test]
    fn should_trim_mixed_indentation() {
        let s = "\t    line1\n\t    line2".to_string();
        let r = unindent(&s);
        assert_eq!(r, "line1\nline2");

        let s = "    \tline1\n    \tline2".to_string();
        let r = unindent(&s);
        assert_eq!(r, "line1\nline2");
    }
}
