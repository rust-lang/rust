// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Prunes things with the #[doc(hidden)] attribute

use astsrv;
use attr_parser;
use doc::ItemUtils;
use doc;
use fold::Fold;
use fold;
use pass::Pass;

pub fn mk_pass() -> Pass {
    Pass {
        name: ~"prune_hidden",
        f: run
    }
}

pub fn run(srv: astsrv::Srv, doc: doc::Doc) -> doc::Doc {
    let fold = Fold {
        ctxt: srv.clone(),
        fold_mod: fold_mod,
        .. fold::default_any_fold(srv)
    };
    (fold.fold_doc)(&fold, doc)
}

fn fold_mod(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::ModDoc
) -> doc::ModDoc {
    let doc = fold::default_any_fold_mod(fold, doc);

    doc::ModDoc {
        items: do doc.items.filtered |ItemTag| {
            !is_hidden(fold.ctxt.clone(), ItemTag.item())
        },
        .. doc
    }
}

fn is_hidden(srv: astsrv::Srv, doc: doc::ItemDoc) -> bool {
    use syntax::ast_map;

    let id = doc.id;
    do astsrv::exec(srv) |ctxt| {
        let attrs = match *ctxt.ast_map.get(&id) {
          ast_map::node_item(item, _) => copy item.attrs,
          _ => ~[]
        };
        attr_parser::parse_hidden(attrs)
    }
}

#[test]
fn should_prune_hidden_items() {
    use core::vec;

    let doc = test::mk_doc(~"#[doc(hidden)] mod a { }");
    assert!(vec::is_empty(doc.cratemod().mods()))
}

#[cfg(test)]
pub mod test {
    use astsrv;
    use doc;
    use extract;
    use prune_hidden_pass::run;

    pub fn mk_doc(source: ~str) -> doc::Doc {
        do astsrv::from_str(copy source) |srv| {
            let doc = extract::from_srv(srv.clone(), ~"");
            run(srv.clone(), doc)
        }
    }
}
