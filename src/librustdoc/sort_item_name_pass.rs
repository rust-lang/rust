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

use doc::ItemUtils;
use doc;
use pass::Pass;
use sort_pass;

pub fn mk_pass() -> Pass {
    fn by_item_name(item1: &doc::ItemTag, item2: &doc::ItemTag) -> bool {
        (*item1).name() <= (*item2).name()
    }
    sort_pass::mk_pass(~"sort_item_name", by_item_name)
}

#[test]
fn test() {
    use astsrv;
    use extract;

    let source = ~"mod z { } fn y() { }";
    do astsrv::from_str(source) |srv| {
        let doc = extract::from_srv(srv.clone(), ~"");
        let doc = (mk_pass().f)(srv.clone(), doc);
        // hidden __std_macros module at the start.
        assert_eq!(doc.cratemod().items[1].name(), ~"y");
        assert_eq!(doc.cratemod().items[2].name(), ~"z");
    }
}
