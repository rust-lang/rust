// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::num;
use std::uint;
use std::hashmap::HashSet;

use syntax::ast;

use clean;
use clean::Item;
use plugins;
use fold;
use fold::DocFolder;

/// Strip items marked `#[doc(hidden)]`
pub fn strip_hidden(crate: clean::Crate) -> plugins::PluginResult {
    struct Stripper;
    impl fold::DocFolder for Stripper {
        fn fold_item(&mut self, i: Item) -> Option<Item> {
            for attr in i.attrs.iter() {
                match attr {
                    &clean::List(~"doc", ref l) => {
                        for innerattr in l.iter() {
                            match innerattr {
                                &clean::Word(ref s) if "hidden" == *s => {
                                    debug2!("found one in strip_hidden; removing");
                                    return None;
                                },
                                _ => (),
                            }
                        }
                    },
                    _ => ()
                }
            }
            self.fold_item_recur(i)
        }
    }
    let mut stripper = Stripper;
    let crate = stripper.fold_crate(crate);
    (crate, None)
}

/// Strip private items from the point of view of a crate or externally from a
/// crate, specified by the `xcrate` flag.
pub fn strip_private(mut crate: clean::Crate) -> plugins::PluginResult {
    // This stripper collects all *retained* nodes.
    struct Stripper<'self>(&'self mut HashSet<ast::NodeId>);
    impl<'self> fold::DocFolder for Stripper<'self> {
        fn fold_item(&mut self, i: Item) -> Option<Item> {
            match i.inner {
                // These items can all get re-exported
                clean::TypedefItem(*) | clean::StaticItem(*) |
                clean::StructItem(*) | clean::EnumItem(*) |
                clean::TraitItem(*) | clean::FunctionItem(*) |
                clean::ViewItemItem(*) | clean::MethodItem(*) |
                clean::ForeignFunctionItem(*) | clean::ForeignStaticItem(*) => {
                    // XXX: re-exported items should get surfaced in the docs as
                    //      well (using the output of resolve analysis)
                    if i.visibility != Some(ast::public) {
                        return None;
                    }
                }

                // These are public-by-default (if the enum was public)
                clean::VariantItem(*) => {
                    if i.visibility == Some(ast::private) {
                        return None;
                    }
                }

                // We show these regardless of whether they're public/private
                // because it's useful to see sometimes
                clean::StructFieldItem(*) => {}

                // handled below
                clean::ModuleItem(*) => {}

                // impls/tymethods have no control over privacy
                clean::ImplItem(*) | clean::TyMethodItem(*) => {}
            }

            let fastreturn = match i.inner {
                // nothing left to do for traits (don't want to filter their
                // methods out, visibility controlled by the trait)
                clean::TraitItem(*) => true,

                // implementations of traits are always public.
                clean::ImplItem(ref imp) if imp.trait_.is_some() => true,

                _ => false,
            };

            let i = if fastreturn {
                self.insert(i.id);
                return Some(i);
            } else {
                self.fold_item_recur(i)
            };

            match i {
                Some(i) => {
                    match i.inner {
                        // emptied modules/impls have no need to exist
                        clean::ModuleItem(ref m) if m.items.len() == 0 => None,
                        clean::ImplItem(ref i) if i.methods.len() == 0 => None,
                        _ => {
                            self.insert(i.id);
                            Some(i)
                        }
                    }
                }
                None => None,
            }
        }
    }

    // This stripper discards all private impls of traits
    struct ImplStripper<'self>(&'self HashSet<ast::NodeId>);
    impl<'self> fold::DocFolder for ImplStripper<'self> {
        fn fold_item(&mut self, i: Item) -> Option<Item> {
            match i.inner {
                clean::ImplItem(ref imp) => {
                    match imp.trait_ {
                        Some(clean::ResolvedPath{ id, _ }) => {
                            if !self.contains(&id) {
                                return None;
                            }
                        }
                        Some(*) | None => {}
                    }
                }
                _ => {}
            }
            self.fold_item_recur(i)
        }
    }

    let mut retained = HashSet::new();
    // First, strip all private items
    {
        let mut stripper = Stripper(&mut retained);
        crate = stripper.fold_crate(crate);
    }

    // Next, strip all private implementations of traits
    {
        let mut stripper = ImplStripper(&retained);
        crate = stripper.fold_crate(crate);
    }
    (crate, None)
}

pub fn unindent_comments(crate: clean::Crate) -> plugins::PluginResult {
    struct CommentCleaner;
    impl fold::DocFolder for CommentCleaner {
        fn fold_item(&mut self, i: Item) -> Option<Item> {
            let mut i = i;
            let mut avec: ~[clean::Attribute] = ~[];
            for attr in i.attrs.iter() {
                match attr {
                    &clean::NameValue(~"doc", ref s) => avec.push(
                        clean::NameValue(~"doc", unindent(*s))),
                    x => avec.push(x.clone())
                }
            }
            i.attrs = avec;
            self.fold_item_recur(i)
        }
    }
    let mut cleaner = CommentCleaner;
    let crate = cleaner.fold_crate(crate);
    (crate, None)
}

pub fn collapse_docs(crate: clean::Crate) -> plugins::PluginResult {
    struct Collapser;
    impl fold::DocFolder for Collapser {
        fn fold_item(&mut self, i: Item) -> Option<Item> {
            let mut docstr = ~"";
            let mut i = i;
            for attr in i.attrs.iter() {
                match *attr {
                    clean::NameValue(~"doc", ref s) => {
                        docstr.push_str(s.clone());
                        docstr.push_char('\n');
                    },
                    _ => ()
                }
            }
            let mut a: ~[clean::Attribute] = i.attrs.iter().filter(|&a| match a {
                &clean::NameValue(~"doc", _) => false,
                _ => true
            }).map(|x| x.clone()).collect();
            if "" != docstr {
                a.push(clean::NameValue(~"doc", docstr));
            }
            i.attrs = a;
            self.fold_item_recur(i)
        }
    }
    let mut collapser = Collapser;
    let crate = collapser.fold_crate(crate);
    (crate, None)
}

pub fn unindent(s: &str) -> ~str {
    let lines = s.any_line_iter().collect::<~[&str]>();
    let mut saw_first_line = false;
    let mut saw_second_line = false;
    let min_indent = do lines.iter().fold(uint::max_value) |min_indent, line| {

        // After we see the first non-whitespace line, look at
        // the line we have. If it is not whitespace, and therefore
        // part of the first paragraph, then ignore the indentation
        // level of the first line
        let ignore_previous_indents =
            saw_first_line &&
            !saw_second_line &&
            !line.is_whitespace();

        let min_indent = if ignore_previous_indents {
            uint::max_value
        } else {
            min_indent
        };

        if saw_first_line {
            saw_second_line = true;
        }

        if line.is_whitespace() {
            min_indent
        } else {
            saw_first_line = true;
            let mut spaces = 0;
            do line.iter().all |char| {
                // Only comparing against space because I wouldn't
                // know what to do with mixed whitespace chars
                if char == ' ' {
                    spaces += 1;
                    true
                } else {
                    false
                }
            };
            num::min(min_indent, spaces)
        }
    };

    match lines {
        [head, .. tail] => {
            let mut unindented = ~[ head.trim() ];
            unindented.push_all(do tail.map |&line| {
                if line.is_whitespace() {
                    line
                } else {
                    assert!(line.len() >= min_indent);
                    line.slice_from(min_indent)
                }
            });
            unindented.connect("\n")
        }
        [] => s.to_owned()
    }
}

#[cfg(test)]
mod unindent_tests {
    use super::unindent;

    #[test]
    fn should_unindent() {
        let s = ~"    line1\n    line2";
        let r = unindent(s);
        assert_eq!(r, ~"line1\nline2");
    }

    #[test]
    fn should_unindent_multiple_paragraphs() {
        let s = ~"    line1\n\n    line2";
        let r = unindent(s);
        assert_eq!(r, ~"line1\n\nline2");
    }

    #[test]
    fn should_leave_multiple_indent_levels() {
        // Line 2 is indented another level beyond the
        // base indentation and should be preserved
        let s = ~"    line1\n\n        line2";
        let r = unindent(s);
        assert_eq!(r, ~"line1\n\n    line2");
    }

    #[test]
    fn should_ignore_first_line_indent() {
        // Thi first line of the first paragraph may not be indented as
        // far due to the way the doc string was written:
        //
        // #[doc = "Start way over here
        //          and continue here"]
        let s = ~"line1\n    line2";
        let r = unindent(s);
        assert_eq!(r, ~"line1\nline2");
    }

    #[test]
    fn should_not_ignore_first_line_indent_in_a_single_line_para() {
        let s = ~"line1\n\n    line2";
        let r = unindent(s);
        assert_eq!(r, ~"line1\n\n    line2");
    }
}
