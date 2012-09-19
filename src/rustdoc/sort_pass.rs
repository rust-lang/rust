//! A general sorting pass

use doc::ItemUtils;
use std::sort;

export item_lteq, mk_pass;

type ItemLtEq = pure fn~(v1: &doc::ItemTag, v2:  &doc::ItemTag) -> bool;

fn mk_pass(name: ~str, +lteq: ItemLtEq) -> Pass {
    {
        name: name,
        f: fn~(move lteq, srv: astsrv::Srv, doc: doc::Doc) -> doc::Doc {
            run(srv, doc, lteq)
        }
    }
}

#[allow(non_implicitly_copyable_typarams)]
fn run(
    _srv: astsrv::Srv,
    doc: doc::Doc,
    lteq: ItemLtEq
) -> doc::Doc {
    let fold = fold::Fold({
        fold_mod: fold_mod,
        .. *fold::default_any_fold(lteq)
    });
    fold.fold_doc(fold, doc)
}

#[allow(non_implicitly_copyable_typarams)]
fn fold_mod(
    fold: fold::Fold<ItemLtEq>,
    doc: doc::ModDoc
) -> doc::ModDoc {
    let doc = fold::default_any_fold_mod(fold, doc);
    doc::ModDoc_({
        items: sort::merge_sort(fold.ctxt, doc.items),
        .. *doc
    })
}

#[test]
fn test() {
    pure fn name_lteq(item1: &doc::ItemTag, item2: &doc::ItemTag) -> bool {
        (*item1).name() <= (*item2).name()
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
    pure fn always_eq(_item1: &doc::ItemTag, _item2: &doc::ItemTag) -> bool {
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
