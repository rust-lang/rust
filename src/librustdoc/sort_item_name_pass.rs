// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Sorts items by name

use astsrv;
use doc::ItemUtils;
use doc;
use extract;
use pass::Pass;
use sort_pass;

pub fn mk_pass() -> Pass {
    pure fn by_item_name(item1: &doc::ItemTag, item2: &doc::ItemTag) -> bool {
        (*item1).name() <= (*item2).name()
    }
    sort_pass::mk_pass(~"sort_item_name", by_item_name)
}

#[test]
fn test() {
    let source = ~"mod z { } fn y() { }";
    do astsrv::from_str(source) |srv| {
        let doc = extract::from_srv(srv, ~"");
        let doc = (mk_pass().f)(srv, doc);
        assert doc.cratemod().items[0].name() == ~"y";
        assert doc.cratemod().items[1].name() == ~"z";
    }
}
