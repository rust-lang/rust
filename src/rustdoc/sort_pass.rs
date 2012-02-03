#[doc = "A general sorting pass"];

import std::sort;

export item_lteq, mk_pass;

type item_lteq = fn~(doc::itemtag, doc::itemtag) -> bool;

fn mk_pass(lteq: item_lteq) -> pass {
    fn~(srv: astsrv::srv, doc: doc::cratedoc) -> doc::cratedoc {
        run(srv, doc, lteq)
    }
}

fn run(
    _srv: astsrv::srv,
    doc: doc::cratedoc,
    lteq: item_lteq
) -> doc::cratedoc {
    let fold = fold::fold({
        fold_mod: fold_mod
        with *fold::default_seq_fold(lteq)
    });
    fold.fold_crate(fold, doc)
}

fn fold_mod(
    fold: fold::fold<item_lteq>,
    doc: doc::moddoc
) -> doc::moddoc {
    let doc = fold::default_seq_fold_mod(fold, doc);
    {
        items: ~sort::merge_sort(fold.ctxt, *doc.items)
        with doc
    }
}

#[test]
fn test() {
    fn name_lteq(item1: doc::itemtag, item2: doc::itemtag) -> bool {
        str::le(item1.name(), item2.name())
    }

    let source = "mod z { mod y { } fn x() { } } mod w { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = mk_pass(name_lteq)(srv, doc);
    assert doc.topmod.mods()[0].name == "w";
    assert doc.topmod.mods()[1].items[0].name() == "x";
    assert doc.topmod.mods()[1].items[1].name() == "y";
    assert doc.topmod.mods()[1].name == "z";
}

#[test]
fn should_be_stable() {
    fn always_eq(_item1: doc::itemtag, _item2: doc::itemtag) -> bool {
        true
    }

    let source = "mod a { mod b { } } mod c { mod d { } }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = mk_pass(always_eq)(srv, doc);
    assert doc.topmod.mods()[0].items[0].name() == "b";
    assert doc.topmod.mods()[1].items[0].name() == "d";
    let doc = mk_pass(always_eq)(srv, doc);
    assert doc.topmod.mods()[0].items[0].name() == "b";
    assert doc.topmod.mods()[1].items[0].name() == "d";
}
