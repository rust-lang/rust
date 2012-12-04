// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Prune things that are private

#[legacy_exports];

export mk_pass;

fn mk_pass() -> Pass {
    {
        name: ~"prune_private",
        f: run
    }
}

fn run(srv: astsrv::Srv, +doc: doc::Doc) -> doc::Doc {
    let fold = fold::Fold({
        fold_mod: fold_mod,
        .. *fold::default_any_fold(srv)
    });
    (fold.fold_doc)(&fold, doc)
}

fn fold_mod(
    fold: &fold::Fold<astsrv::Srv>,
    +doc: doc::ModDoc
) -> doc::ModDoc {
    let doc = fold::default_any_fold_mod(fold, doc);

    doc::ModDoc_({
        items: do doc.items.filter |ItemTag| {
            is_visible(fold.ctxt, ItemTag.item())
        },
        .. *doc
    })
}

fn is_visible(srv: astsrv::Srv, doc: doc::ItemDoc) -> bool {
    use syntax::ast_map;
    use syntax::ast;

    let id = doc.id;

    do astsrv::exec(srv) |ctxt| {
        match ctxt.ast_map.get(id) {
            ast_map::node_item(item, _) => {
                item.vis == ast::public
            }
            _ => core::util::unreachable()
        }
    }
}

#[test]
fn should_prune_items_without_pub_modifier() {
    let doc = test::mk_doc(~"mod a { }");
    assert vec::is_empty(doc.cratemod().mods());
}

#[cfg(test)]
mod test {
    pub fn mk_doc(source: ~str) -> doc::Doc {
        do astsrv::from_str(source) |srv| {
            let doc = extract::from_srv(srv, ~"");
            run(srv, doc)
        }
    }
}

