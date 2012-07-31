//! Sorts items by type

import doc::item_utils;

export mk_pass;

fn mk_pass() -> pass {
    do sort_pass::mk_pass(~"sort_item_type") |item1, item2| {
        fn score(item: doc::itemtag) -> int {
            alt item {
              doc::consttag(_) { 0 }
              doc::tytag(_) { 1 }
              doc::enumtag(_) { 2 }
              doc::traittag(_) { 3 }
              doc::impltag(_) { 4 }
              doc::fntag(_) { 5 }
              doc::modtag(_) { 6 }
              doc::nmodtag(_) { 7 }
            }
        }

        score(item1) <= score(item2)
    }
}

#[test]
fn test() {
    let source =
        ~"mod imod { } \
         extern mod inmod { } \
         const iconst: int = 0; \
         fn ifn() { } \
         enum ienum { ivar } \
         trait itrait { fn a(); } \
         impl iimpl for int { fn a() { } } \
         type itype = int;";
    do astsrv::from_str(source) |srv| {
        let doc = extract::from_srv(srv, ~"");
        let doc = mk_pass().f(srv, doc);
        assert doc.cratemod().items[0].name() == ~"iconst";
        assert doc.cratemod().items[1].name() == ~"itype";
        assert doc.cratemod().items[2].name() == ~"ienum";
        assert doc.cratemod().items[3].name() == ~"itrait";
        assert doc.cratemod().items[4].name() == ~"iimpl";
        assert doc.cratemod().items[5].name() == ~"ifn";
        assert doc.cratemod().items[6].name() == ~"imod";
        assert doc.cratemod().items[7].name() == ~"inmod";
    }
}
