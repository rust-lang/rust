// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use clean::{self, Item};
use plugins;
use fold;
use fold::DocFolder;

pub fn collapse_docs(krate: clean::Crate) -> plugins::PluginResult {
    Collapser.fold_crate(krate)
}

struct Collapser;

impl fold::DocFolder for Collapser {
    fn fold_item(&mut self, mut i: Item) -> Option<Item> {
        i.attrs.collapse_doc_comments();
        self.fold_item_recur(i)
    }
}

impl clean::Attributes {
    pub fn collapse_doc_comments(&mut self) {
        let mut doc_string = self.doc_strings.join("\n");
        if doc_string.is_empty() {
            self.doc_strings = vec![];
        } else {
            // FIXME(eddyb) Is this still needed?
            doc_string.push('\n');
            self.doc_strings = vec![doc_string];
        }
    }
}
