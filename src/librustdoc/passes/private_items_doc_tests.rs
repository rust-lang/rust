// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use clean::*;

use core::DocContext;
use fold::DocFolder;

use passes::{look_for_tests, Pass};

pub const CHECK_PRIVATE_ITEMS_DOC_TESTS: Pass =
    Pass::early("check-private-items-doc-tests", check_private_items_doc_tests,
                "check private items doc tests");

struct PrivateItemDocTestLinter<'a, 'tcx: 'a, 'rcx: 'a> {
    cx: &'a DocContext<'a, 'tcx, 'rcx>,
}

impl<'a, 'tcx, 'rcx> PrivateItemDocTestLinter<'a, 'tcx, 'rcx> {
    fn new(cx: &'a DocContext<'a, 'tcx, 'rcx>) -> Self {
        PrivateItemDocTestLinter {
            cx,
        }
    }
}

pub fn check_private_items_doc_tests(krate: Crate, cx: &DocContext) -> Crate {
    let mut coll = PrivateItemDocTestLinter::new(cx);

    coll.fold_crate(krate)
}

impl<'a, 'tcx, 'rcx> DocFolder for PrivateItemDocTestLinter<'a, 'tcx, 'rcx> {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        let cx = self.cx;
        let dox = item.attrs.collapsed_doc_value().unwrap_or_else(String::new);

        look_for_tests(&cx, &dox, &item, false);

        self.fold_item_recur(item)
    }
}
