//! Prunes things with the #[doc(hidden)] attribute

use doc::item_utils;
use std::map::HashMap;
export mk_pass;

fn mk_pass() -> pass {
    {
        name: ~"prune_hidden",
        f: run
    }
}

fn run(srv: astsrv::srv, doc: doc::doc) -> doc::doc {
    let fold = fold::fold({
        fold_mod: fold_mod,
        .. *fold::default_any_fold(srv)
    });
    fold.fold_doc(fold, doc)
}

fn fold_mod(
    fold: fold::fold<astsrv::srv>,
    doc: doc::moddoc
) -> doc::moddoc {
    let doc = fold::default_any_fold_mod(fold, doc);

    doc::moddoc_({
        items: vec::filter(doc.items, |itemtag| {
            !is_hidden(fold.ctxt, itemtag.item())
        }),
        .. *doc
    })
}

fn is_hidden(srv: astsrv::srv, doc: doc::itemdoc) -> bool {
    use syntax::ast_map;

    let id = doc.id;
    do astsrv::exec(srv) |ctxt| {
        let attrs = match ctxt.ast_map.get(id) {
          ast_map::node_item(item, _) => item.attrs,
          _ => ~[]
        };
        attr_parser::parse_hidden(attrs)
    }
}

#[test]
fn should_prune_hidden_items() {
    let doc = test::mk_doc(~"#[doc(hidden)] mod a { }");
    assert vec::is_empty(doc.cratemod().mods())
}

#[cfg(test)]
mod test {
    fn mk_doc(source: ~str) -> doc::doc {
        do astsrv::from_str(source) |srv| {
            let doc = extract::from_srv(srv, ~"");
            run(srv, doc)
        }
    }
}
