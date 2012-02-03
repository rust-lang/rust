#[doc = "Sorts items by name"];

export mk_pass;

fn mk_pass() -> pass {
    sort_pass::mk_pass { |item1, item2|
        str::le(item1.name(), item2.name())
    }
}

#[test]
fn test() {
    let source = "mod z { } fn y() { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = mk_pass()(srv, doc);
    assert doc.topmod.items[0].name() == "y";
    assert doc.topmod.items[1].name() == "z";
}
