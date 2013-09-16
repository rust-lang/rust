// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std;
use clean;
use syntax::ast;
use clean::Item;
use plugins;
use fold;
use fold::DocFolder;

/// A sample pass showing the minimum required work for a plugin.
pub fn noop(crate: clean::Crate) -> plugins::PluginResult {
    (crate, None)
}

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
                                    info!("found one in strip_hidden; removing");
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

pub fn clean_comments(crate: clean::Crate) -> plugins::PluginResult {
    struct CommentCleaner;
    impl fold::DocFolder for CommentCleaner {
        fn fold_item(&mut self, i: Item) -> Option<Item> {
            let mut i = i;
            let mut avec: ~[clean::Attribute] = ~[];
            for attr in i.attrs.iter() {
                match attr {
                    &clean::NameValue(~"doc", ref s) => avec.push(
                        clean::NameValue(~"doc", clean_comment_body(s.clone()))),
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

pub fn collapse_privacy(crate: clean::Crate) -> plugins::PluginResult {
    struct PrivacyCollapser {
        stack: ~[clean::Visibility]
    }
    impl fold::DocFolder for PrivacyCollapser {
        fn fold_item(&mut self, mut i: Item) -> Option<Item> {
            if i.visibility.is_some() {
                if i.visibility == Some(ast::inherited) {
                    i.visibility = Some(self.stack.last().clone());
                } else {
                    self.stack.push(i.visibility.clone().unwrap());
                }
            }
            self.fold_item_recur(i)
        }
    }
    let mut privacy = PrivacyCollapser { stack: ~[] };
    let crate = privacy.fold_crate(crate);
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
                a.push(clean::NameValue(~"doc", docstr.trim().to_owned()));
            }
            i.attrs = a;
            self.fold_item_recur(i)
        }
    }
    let mut collapser = Collapser;
    let crate = collapser.fold_crate(crate);
    (crate, None)
}

//Utility
enum CleanCommentStates {
    Collect,
    Strip,
    Stripped,
}

/// Returns the index of the last character all strings have common in their
/// prefix.
fn longest_common_prefix(s: ~[~str]) -> uint {
    // find the longest common prefix

    debug!("lcp: looking into %?", s);
    // index of the last character all the strings share
    let mut index = 0u;

    if s.len() <= 1 {
        return 0;
    }

    // whether one of the strings has been exhausted of characters yet
    let mut exhausted = false;

    // character iterators for all the lines
    let mut lines = s.iter().filter(|x| x.len() != 0).map(|x| x.iter()).to_owned_vec();

    'outer: loop {
        // because you can't label a while loop
        if exhausted == true {
            break;
        }
        debug!("lcp: index %u", index);
        let mut lines = lines.mut_iter();
        let ch = match lines.next().unwrap().next() {
            Some(c) => c,
            None => { exhausted = true; loop },
        };
        debug!("looking for char %c", ch);
        for line in lines {
            match line.next() {
                Some(c) => if c == ch { loop } else { exhausted = true; loop 'outer },
                None => { exhausted = true; loop 'outer }
            }
        }
        index += 1;
    }

    debug!("lcp: last index %u", index);
    index
}

fn clean_comment_body(s: ~str) -> ~str {
    // FIXME #31: lots of copies in here.
    let lines = s.line_iter().to_owned_vec();
    match lines.len() {
        0 => return ~"",
        1 => return lines[0].slice_from(2).trim().to_owned(),
        _ => (),
    }

    let mut ol = std::vec::with_capacity(lines.len());
    for line in lines.clone().move_iter() {
        // replace meaningless things with a single newline
        match line {
            x if ["/**", "/*!", "///", "//!", "*/"].contains(&x.trim()) => ol.push(~""),
            x if x.trim() == "" => ol.push(~""),
            x => ol.push(x.to_owned())
        }
    }
    let li = longest_common_prefix(ol.clone());

    let x = ol.iter()
         .filter(|x| { debug!("cleaning line: %s", **x); true })
         .map(|x| if x.len() == 0 { ~"" } else { x.slice_chars(li, x.char_len()).to_owned() })
         .to_owned_vec().connect("\n");
    x.trim().to_owned()
}
