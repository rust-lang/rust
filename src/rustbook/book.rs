// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Basic data structures for representing a book.

use std::io::prelude::*;
use std::io::BufReader;
use std::iter;
use std::path::{Path, PathBuf};

pub struct BookItem {
    pub title: String,
    pub path: PathBuf,
    pub path_to_root: PathBuf,
    pub children: Vec<BookItem>,
}

pub struct Book {
    pub chapters: Vec<BookItem>,
}

/// A depth-first iterator over a book.
pub struct BookItems<'a> {
    cur_items: &'a [BookItem],
    cur_idx: usize,
    stack: Vec<(&'a [BookItem], usize)>,
}

impl<'a> Iterator for BookItems<'a> {
    type Item = (String, &'a BookItem);

    fn next(&mut self) -> Option<(String, &'a BookItem)> {
        loop {
            if self.cur_idx >= self.cur_items.len() {
                match self.stack.pop() {
                    None => return None,
                    Some((parent_items, parent_idx)) => {
                        self.cur_items = parent_items;
                        self.cur_idx = parent_idx + 1;
                    }
                }
            } else {
                let cur = self.cur_items.get(self.cur_idx).unwrap();

                let mut section = String::from("");
                for &(_, idx) in &self.stack {
                    section.push_str(&format!("{}", idx + 1));
                    section.push('.');
                }
                section.push_str(&format!("{}", self.cur_idx + 1));
                section.push('.');

                self.stack.push((self.cur_items, self.cur_idx));
                self.cur_items = &cur.children[..];
                self.cur_idx = 0;
                return Some((section, cur))
            }
        }
    }
}

impl Book {
    pub fn iter(&self) -> BookItems {
        BookItems {
            cur_items: &self.chapters[..],
            cur_idx: 0,
            stack: Vec::new(),
        }
    }
}

/// Construct a book by parsing a summary (markdown table of contents).
pub fn parse_summary(input: &mut Read, src: &Path) -> Result<Book, Vec<String>> {
    fn collapse(stack: &mut Vec<BookItem>,
                top_items: &mut Vec<BookItem>,
                to_level: usize) {
        loop {
            if stack.len() < to_level { return }
            if stack.len() == 1 {
                top_items.push(stack.pop().unwrap());
                return;
            }

            let tip = stack.pop().unwrap();
            let last = stack.len() - 1;
            stack[last].children.push(tip);
        }
    }

    let mut top_items = vec!();
    let mut stack = vec!();
    let mut errors = vec!();

    // always include the introduction
    top_items.push(BookItem {
        title: String::from("Introduction"),
        path: PathBuf::from("README.md"),
        path_to_root: PathBuf::from(""),
        children: vec!(),
    });

    for line_result in BufReader::new(input).lines() {
        let line = match line_result {
            Ok(line) => line,
            Err(err) => {
                errors.push(String::from(err));
                return Err(errors);
            }
        };

        let star_idx = match line.find("*") { Some(i) => i, None => continue };

        let start_bracket = star_idx + line[star_idx..].find("[").unwrap();
        let end_bracket = start_bracket + line[start_bracket..].find("](").unwrap();
        let start_paren = end_bracket + 1;
        let end_paren = start_paren + line[start_paren..].find(")").unwrap();

        let given_path = &line[start_paren + 1 .. end_paren];
        let title = String::from(line[start_bracket + 1..end_bracket]);
        let indent = &line[..star_idx];

        let path_from_root = match src.join(given_path).relative_from(src) {
            Some(p) => p.to_path_buf(),
            None => {
                errors.push(format!("paths in SUMMARY.md must be relative, \
                                     but path '{}' for section '{}' is not.",
                                     given_path, title));
                PathBuf::new()
            }
        };
        let path_to_root = PathBuf::from(&iter::repeat("../")
                                         .take(path_from_root.components().count() - 1)
                                         .collect::<String>());
        let item = BookItem {
            title: title,
            path: path_from_root,
            path_to_root: path_to_root,
            children: vec!(),
        };
        let level = indent.chars().map(|c| -> usize {
            match c {
                ' ' => 1,
                '\t' => 4,
                _ => unreachable!()
            }
        }).sum::<usize>() / 4 + 1;

        if level > stack.len() + 1 {
            errors.push(format!("section '{}' is indented too deeply; \
                                 found {}, expected {} or less",
                                item.title, level, stack.len() + 1));
        } else if level <= stack.len() {
            collapse(&mut stack, &mut top_items, level);
        }
        stack.push(item)
    }

    if errors.is_empty() {
        collapse(&mut stack, &mut top_items, 1);
        Ok(Book { chapters: top_items })
    } else {
        Err(errors)
    }
}
