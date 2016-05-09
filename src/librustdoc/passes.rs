// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::DefId;
use rustc::middle::privacy::AccessLevels;
use rustc::util::nodemap::DefIdSet;
use std::cmp;
use std::string::String;
use std::usize;

use clean::{self, Attributes, GetDefId};
use clean::Item;
use plugins;
use fold;
use fold::DocFolder;
use fold::FoldItem::Strip;

/// Strip items marked `#[doc(hidden)]`
pub fn strip_hidden(krate: clean::Crate) -> plugins::PluginResult {
    let mut retained = DefIdSet();

    // strip all #[doc(hidden)] items
    let krate = {
        struct Stripper<'a> {
            retained: &'a mut DefIdSet
        }
        impl<'a> fold::DocFolder for Stripper<'a> {
            fn fold_item(&mut self, i: Item) -> Option<Item> {
                if i.attrs.list("doc").has_word("hidden") {
                    debug!("found one in strip_hidden; removing");
                    // use a dedicated hidden item for given item type if any
                    match i.inner {
                        clean::StructFieldItem(..) | clean::ModuleItem(..) => {
                            return Strip(i).fold()
                        }
                        _ => return None,
                    }
                } else {
                    self.retained.insert(i.def_id);
                }
                self.fold_item_recur(i)
            }
        }
        let mut stripper = Stripper{ retained: &mut retained };
        stripper.fold_crate(krate)
    };

    // strip all impls referencing stripped items
    let mut stripper = ImplStripper { retained: &retained };
    stripper.fold_crate(krate)
}

/// Strip private items from the point of view of a crate or externally from a
/// crate, specified by the `xcrate` flag.
pub fn strip_private(mut krate: clean::Crate) -> plugins::PluginResult {
    // This stripper collects all *retained* nodes.
    let mut retained = DefIdSet();
    let access_levels = krate.access_levels.clone();

    // strip all private items
    {
        let mut stripper = Stripper {
            retained: &mut retained,
            access_levels: &access_levels,
        };
        krate = ImportStripper.fold_crate(stripper.fold_crate(krate));
    }

    // strip all impls referencing private items
    let mut stripper = ImplStripper { retained: &retained };
    stripper.fold_crate(krate)
}

struct Stripper<'a> {
    retained: &'a mut DefIdSet,
    access_levels: &'a AccessLevels<DefId>,
}

impl<'a> fold::DocFolder for Stripper<'a> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        match i.inner {
            clean::StrippedItem(..) => return Some(i),
            // These items can all get re-exported
            clean::TypedefItem(..) | clean::StaticItem(..) |
            clean::StructItem(..) | clean::EnumItem(..) |
            clean::TraitItem(..) | clean::FunctionItem(..) |
            clean::VariantItem(..) | clean::MethodItem(..) |
            clean::ForeignFunctionItem(..) | clean::ForeignStaticItem(..) |
            clean::ConstantItem(..) => {
                if i.def_id.is_local() {
                    if !self.access_levels.is_exported(i.def_id) {
                        return None;
                    }
                }
            }

            clean::StructFieldItem(..) => {
                if i.visibility != Some(clean::Public) {
                    return Strip(i).fold();
                }
            }

            clean::ModuleItem(..) => {
                if i.def_id.is_local() && i.visibility != Some(clean::Public) {
                    return Strip(self.fold_item_recur(i).unwrap()).fold()
                }
            }

            // trait impls for private items should be stripped
            clean::ImplItem(clean::Impl{
                for_: clean::ResolvedPath{ did, is_generic, .. }, ..
            }) => {
                if did.is_local() && !is_generic && !self.access_levels.is_exported(did) {
                    return None;
                }
            }
            // handled in the `strip-priv-imports` pass
            clean::ExternCrateItem(..) | clean::ImportItem(..) => {}

            clean::DefaultImplItem(..) | clean::ImplItem(..) => {}

            // tymethods/macros have no control over privacy
            clean::MacroItem(..) | clean::TyMethodItem(..) => {}

            // Primitives are never stripped
            clean::PrimitiveItem(..) => {}

            // Associated consts and types are never stripped
            clean::AssociatedConstItem(..) |
            clean::AssociatedTypeItem(..) => {}
        }

        let fastreturn = match i.inner {
            // nothing left to do for traits (don't want to filter their
            // methods out, visibility controlled by the trait)
            clean::TraitItem(..) => true,

            // implementations of traits are always public.
            clean::ImplItem(ref imp) if imp.trait_.is_some() => true,
            // Struct variant fields have inherited visibility
            clean::VariantItem(clean::Variant {
                kind: clean::StructVariant(..)
            }) => true,
            _ => false,
        };

        let i = if fastreturn {
            self.retained.insert(i.def_id);
            return Some(i);
        } else {
            self.fold_item_recur(i)
        };

        i.and_then(|i| {
            match i.inner {
                // emptied modules/impls have no need to exist
                clean::ModuleItem(ref m)
                    if m.items.is_empty() &&
                       i.doc_value().is_none() => None,
                clean::ImplItem(ref i) if i.items.is_empty() => None,
                _ => {
                    self.retained.insert(i.def_id);
                    Some(i)
                }
            }
        })
    }
}

// This stripper discards all impls which reference stripped items
struct ImplStripper<'a> {
    retained: &'a DefIdSet
}

impl<'a> fold::DocFolder for ImplStripper<'a> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        if let clean::ImplItem(ref imp) = i.inner {
            if let Some(did) = imp.for_.def_id() {
                if did.is_local() && !imp.for_.is_generic() &&
                    !self.retained.contains(&did)
                {
                    return None;
                }
            }
            if let Some(did) = imp.trait_.def_id() {
                if did.is_local() && !self.retained.contains(&did) {
                    return None;
                }
            }
        }
        self.fold_item_recur(i)
    }
}

// This stripper discards all private import statements (`use`, `extern crate`)
struct ImportStripper;
impl fold::DocFolder for ImportStripper {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        match i.inner {
            clean::ExternCrateItem(..) |
            clean::ImportItem(..) if i.visibility != Some(clean::Public) => None,
            _ => self.fold_item_recur(i)
        }
    }
}

pub fn strip_priv_imports(krate: clean::Crate)  -> plugins::PluginResult {
    ImportStripper.fold_crate(krate)
}

pub fn unindent_comments(krate: clean::Crate) -> plugins::PluginResult {
    struct CommentCleaner;
    impl fold::DocFolder for CommentCleaner {
        fn fold_item(&mut self, mut i: Item) -> Option<Item> {
            let mut avec: Vec<clean::Attribute> = Vec::new();
            for attr in &i.attrs {
                match attr {
                    &clean::NameValue(ref x, ref s)
                            if "doc" == *x => {
                        avec.push(clean::NameValue("doc".to_string(),
                                                   unindent(s)))
                    }
                    x => avec.push(x.clone())
                }
            }
            i.attrs = avec;
            self.fold_item_recur(i)
        }
    }
    let mut cleaner = CommentCleaner;
    let krate = cleaner.fold_crate(krate);
    krate
}

pub fn collapse_docs(krate: clean::Crate) -> plugins::PluginResult {
    struct Collapser;
    impl fold::DocFolder for Collapser {
        fn fold_item(&mut self, mut i: Item) -> Option<Item> {
            let mut docstr = String::new();
            for attr in &i.attrs {
                if let clean::NameValue(ref x, ref s) = *attr {
                    if "doc" == *x {
                        docstr.push_str(s);
                        docstr.push('\n');
                    }
                }
            }
            let mut a: Vec<clean::Attribute> = i.attrs.iter().filter(|&a| match a {
                &clean::NameValue(ref x, _) if "doc" == *x => false,
                _ => true
            }).cloned().collect();
            if !docstr.is_empty() {
                a.push(clean::NameValue("doc".to_string(), docstr));
            }
            i.attrs = a;
            self.fold_item_recur(i)
        }
    }
    let mut collapser = Collapser;
    let krate = collapser.fold_crate(krate);
    krate
}

pub fn unindent(s: &str) -> String {
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
