// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Sorts items by type

use doc;
use pass::Pass;
use sort_pass;

pub fn mk_pass() -> Pass {
    fn by_score(item1: &doc::ItemTag, item2: &doc::ItemTag) -> bool {
        fn score(item: &doc::ItemTag) -> int {
            match *item {
              doc::ConstTag(_) => 0,
              doc::TyTag(_) => 1,
              doc::EnumTag(_) => 2,
              doc::StructTag(_) => 3,
              doc::TraitTag(_) => 4,
              doc::ImplTag(_) => 5,
              doc::FnTag(_) => 6,
              doc::ModTag(_) => 7,
              doc::NmodTag(_) => 8
            }
        }

        score(item1) <= score(item2)
    }

    sort_pass::mk_pass(~"sort_item_type", by_score)
}

#[test]
fn test() {
    use astsrv;
    use extract;

    let source =
        ~"mod imod { } \
         static iconst: int = 0; \
         fn ifn() { } \
         enum ienum { ivar } \
         trait itrait { fn a(); } \
         pub impl int { fn a() { } } \
         type itype = int; \
         struct istruct { f: () }";
    do astsrv::from_str(source) |srv| {
        let doc = extract::from_srv(srv.clone(), ~"");
        let doc = (mk_pass().f)(srv.clone(), doc);
        assert!(doc.cratemod().items[0].name() == ~"iconst");
        assert!(doc.cratemod().items[1].name() == ~"itype");
        assert!(doc.cratemod().items[2].name() == ~"ienum");
        assert!(doc.cratemod().items[3].name() == ~"istruct");
        assert!(doc.cratemod().items[4].name() == ~"itrait");
        assert!(doc.cratemod().items[5].name() == ~"__extensions__");
        assert!(doc.cratemod().items[6].name() == ~"ifn");
        assert!(doc.cratemod().items[7].name() == ~"imod");
    }
}
