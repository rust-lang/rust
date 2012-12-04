//! Sorts items by name

use doc::ItemUtils;

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
