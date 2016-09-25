// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::string::String;

use clean::{self, Item};
use plugins;
use fold;
use fold::DocFolder;

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
