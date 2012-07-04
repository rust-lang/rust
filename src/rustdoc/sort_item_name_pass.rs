//! Sorts items by name

export mk_pass;

fn mk_pass() -> pass {
    sort_pass::mk_pass("sort_item_name", |item1, item2| {
        str::le(item1.name(), item2.name())
    })
}

#[test]
fn test() {
    let source = "mod z { } fn y() { }";
    do astsrv::from_str(source) |srv| {
        let doc = extract::from_srv(srv, "");
        let doc = mk_pass().f(srv, doc);
        assert doc.cratemod().items[0].name() == "y";
        assert doc.cratemod().items[1].name() == "z";
    }
}
