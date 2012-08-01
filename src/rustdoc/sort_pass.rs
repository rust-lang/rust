//! A general sorting pass

import doc::item_utils;
import std::sort;

export item_lteq, mk_pass;

type item_lteq = fn~(doc::itemtag, doc::itemtag) -> bool;

fn mk_pass(name: ~str, +lteq: item_lteq) -> pass {
    {
        name: name,
        f: fn~(srv: astsrv::srv, doc: doc::doc) -> doc::doc {
            run(srv, doc, lteq)
        }
    }
}

#[allow(non_implicitly_copyable_typarams)]
fn run(
    _srv: astsrv::srv,
    doc: doc::doc,
    lteq: item_lteq
) -> doc::doc {
    let fold = fold::fold({
        fold_mod: fold_mod
        with *fold::default_any_fold(lteq)
    });
    fold.fold_doc(fold, doc)
}

#[allow(non_implicitly_copyable_typarams)]
fn fold_mod(
    fold: fold::fold<item_lteq>,
    doc: doc::moddoc
) -> doc::moddoc {
    let doc = fold::default_any_fold_mod(fold, doc);
    doc::moddoc_({
        items: sort::merge_sort(fold.ctxt, doc.items)
        with *doc
    })
}

#[test]
fn test() {
    fn name_lteq(item1: doc::itemtag, item2: doc::itemtag) -> bool {
        str::le(item1.name(), item2.name())
    }

    let source = ~"mod z { mod y { } fn x() { } } mod w { }";
    do astsrv::from_str(source) |srv| {
        let doc = extract::from_srv(srv, ~"");
        let doc = mk_pass(~"", name_lteq).f(srv, doc);
        assert doc.cratemod().mods()[0].name() == ~"w";
        assert doc.cratemod().mods()[1].items[0].name() == ~"x";
        assert doc.cratemod().mods()[1].items[1].name() == ~"y";
        assert doc.cratemod().mods()[1].name() == ~"z";
    }
}

#[test]
fn should_be_stable() {
    fn always_eq(_item1: doc::itemtag, _item2: doc::itemtag) -> bool {
        true
    }

    let source = ~"mod a { mod b { } } mod c { mod d { } }";
    do astsrv::from_str(source) |srv| {
        let doc = extract::from_srv(srv, ~"");
        let doc = mk_pass(~"", always_eq).f(srv, doc);
        assert doc.cratemod().mods()[0].items[0].name() == ~"b";
        assert doc.cratemod().mods()[1].items[0].name() == ~"d";
        let doc = mk_pass(~"", always_eq).f(srv, doc);
        assert doc.cratemod().mods()[0].items[0].name() == ~"b";
        assert doc.cratemod().mods()[1].items[0].name() == ~"d";
    }
}
